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


FIXED_HADAMARD_BANK_VERSION = "fixed-hadamard-64-v1"


def fixed_hadamard_knot_noise(
    noise_sigma: Union[Sequence[float], torch.Tensor],
    *,
    num_knots: int = 8,
    action_dim: int = 2,
    radii: Sequence[float] = (0.10, 0.30),
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the frozen 64-candidate center-neighborhood design.

    The ordering mirrors the existing MPPI layout: candidate zero is the exact
    center, candidate one is a deterministic extra point, and the remaining 62
    candidates are antithetic pairs.  The 16 Sylvester-Hadamard rows span the
    complete 8x2 knot space.  ``radii`` are measured in source-MPPI sigma units.

    This bank is intentionally independent of the controller RNG.  It is used
    only when explicitly requested; Gaussian sampling remains the default.
    """
    dimension = int(num_knots) * int(action_dim)
    if dimension != 16:
        raise ValueError("fixed-hadamard-64-v1 requires an 8x2 knot space")
    if len(radii) != 2 or not 0 < float(radii[0]) < float(radii[1]):
        raise ValueError("radii must contain two increasing positive values")
    sigma = torch.as_tensor(noise_sigma, dtype=dtype, device=device).reshape(action_dim)
    if torch.any(sigma <= 0):
        raise ValueError("noise_sigma entries must be positive")

    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    directions = matrix.reshape(dimension, num_knots, action_dim)
    scaled_directions = directions * sigma.reshape(1, 1, action_dim)
    inner, outer = (float(radii[0]), float(radii[1]))

    values = [torch.zeros_like(scaled_directions[0])]
    # One fixed extra point preserves the 0 + extra + 31 antithetic-pair
    # structure used by the 64-sample research MPPI wrapper.
    values.append(outer * scaled_directions[-1])
    for direction in scaled_directions:
        values.extend((inner * direction, -inner * direction))
    for direction in scaled_directions[:-1]:
        values.extend((outer * direction, -outer * direction))
    result = torch.stack(values)
    if result.shape != (64, num_knots, action_dim):
        raise AssertionError("fixed Hadamard bank construction produced the wrong shape")
    return result


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
    sampling_mode: str = "gaussian"
    fixed_noise_radii: Tuple[float, float] = (0.10, 0.30)
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
        if self.sampling_mode not in ("gaussian", "fixed_hadamard_64"):
            raise ValueError(
                "sampling_mode must be 'gaussian' or 'fixed_hadamard_64'"
            )
        if self.sampling_mode == "fixed_hadamard_64" and self.num_samples != 64:
            raise ValueError("fixed_hadamard_64 requires num_samples=64")
        if (
            len(self.fixed_noise_radii) != 2
            or not 0 < self.fixed_noise_radii[0] < self.fixed_noise_radii[1]
        ):
            raise ValueError(
                "fixed_noise_radii must contain two increasing positive values"
            )


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

    def _sample_knot_noise(self) -> torch.Tensor:
        if self.params.sampling_mode == "fixed_hadamard_64":
            return fixed_hadamard_knot_noise(
                self._noise_sigma,
                num_knots=self.params.num_knots,
                action_dim=self.params.action_dim,
                radii=self.params.fixed_noise_radii,
                device=self.device,
            )
        noise = torch.randn(
            self.params.num_samples,
            self.params.num_knots,
            self.params.action_dim,
            generator=self._generator,
            device=self.device,
        ) * self._noise_sigma
        noise[0].zero_()  # Always retain the current mean as a candidate.
        return noise

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

    def _prepare_reference_batch(self, reference, batch_size: int) -> torch.Tensor:
        reference = torch.as_tensor(
            reference, dtype=torch.float32, device=self.device
        )
        if reference.ndim != 3 or reference.shape[0] != batch_size:
            raise ValueError("reference must have shape [B,50,4/5] or [B,51,4/5]")
        if reference.shape[2] not in (4, 5):
            raise ValueError("reference must have 4 or 5 state channels")
        if reference.shape[1] == self.params.horizon + 1:
            reference = reference[:, 1:]
        if reference.shape[1] != self.params.horizon:
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

    def context_batch_cost_components(
        self,
        trajectory: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        current_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return costs for B aligned contexts with one action sequence each."""
        batch_size = action.shape[0]
        if trajectory.shape != (
            batch_size,
            self.params.horizon,
            self.params.state_dim,
        ):
            raise ValueError("rollout backend returned an invalid context-batch shape")
        if reference.shape[:2] != (batch_size, self.params.horizon):
            raise ValueError("reference must align with the context batch")
        if current_action.shape != (batch_size, self.params.action_dim):
            raise ValueError("current_action must align with the context batch")
        position_error = (
            trajectory[..., 0:2] - reference[..., 0:2]
        ).square().sum(dim=-1)
        yaw_error = self._wrapped_angle_difference(
            trajectory[..., 2], reference[..., 2]
        ).square()
        vx_error = (trajectory[..., 3] - reference[..., 3]).square()
        components = {
            "position": self.cost_weights.position * position_error.sum(dim=1),
            "yaw": self.cost_weights.yaw * yaw_error.sum(dim=1),
            "vx": self.cost_weights.vx * vx_error.sum(dim=1),
        }
        if reference.shape[2] == 5 and self.cost_weights.yawrate != 0:
            components["yawrate"] = self.cost_weights.yawrate * (
                trajectory[..., 4] - reference[..., 4]
            ).square().sum(dim=1)
        previous = torch.cat((current_action[:, None], action[:, :-1]), dim=1)
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

    @torch.no_grad()
    def evaluate_action_sequences(
        self,
        initial_state,
        current_action,
        history,
        reference,
        action_sequences,
    ) -> Dict[str, object]:
        """Roll out and score deterministic action sequences without MPPI updates.

        This is the model-space primitive used by a baseline-preserving guard.
        It deliberately does not sample, consume the controller RNG, or mutate a
        :class:`TorchMPPIRunningState`.
        """
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
        action_sequences = torch.as_tensor(
            action_sequences, dtype=torch.float32, device=self.device
        )
        if action_sequences.ndim == 2:
            action_sequences = action_sequences.unsqueeze(0)
        expected = (
            self.params.horizon,
            self.params.action_dim,
        )
        if action_sequences.ndim != 3 or tuple(action_sequences.shape[1:]) != expected:
            raise ValueError("action_sequences must have shape [N,50,2] or [50,2]")
        action_sequences = torch.clamp(
            action_sequences, self._action_min, self._action_max
        )
        trajectories = self.rollout_backend(
            history, initial_state, current_action, action_sequences
        ).to(self.device)
        components = self.trajectory_cost_components(
            trajectories, action_sequences, reference, current_action
        )
        cost = sum(components.values())
        return {
            "action_sequences": action_sequences.detach(),
            "trajectories": trajectories.detach(),
            "cost": cost.detach(),
            "cost_components": {
                name: value.detach() for name, value in components.items()
            },
        }

    @torch.no_grad()
    def evaluate_context_action_sequences(
        self,
        initial_state,
        current_action,
        history,
        reference,
        action_sequences,
    ) -> Dict[str, object]:
        """Roll out B contexts with exactly one aligned action sequence each.

        The established ``evaluate_action_sequences`` path remains one physical
        context with N candidates.  This explicit sibling path is B independent
        physical contexts with one candidate per context.
        """
        if not hasattr(self.rollout_backend, "evaluate_context_batch"):
            raise TypeError("rollout backend does not support aligned context batches")
        initial_state = torch.as_tensor(
            initial_state, dtype=torch.float32, device=self.device
        )
        if initial_state.ndim != 2 or initial_state.shape[1] != self.params.state_dim:
            raise ValueError("initial_state must have shape [B,5]")
        batch_size = initial_state.shape[0]
        if batch_size < 1:
            raise ValueError("context batch must be nonempty")
        current_action = torch.as_tensor(
            current_action, dtype=torch.float32, device=self.device
        )
        if current_action.shape != (batch_size, self.params.action_dim):
            raise ValueError("current_action must have shape [B,2]")
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        if history.shape != (batch_size, self.params.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        reference = self._prepare_reference_batch(reference, batch_size)
        action_sequences = torch.as_tensor(
            action_sequences, dtype=torch.float32, device=self.device
        )
        if action_sequences.shape != (
            batch_size,
            self.params.horizon,
            self.params.action_dim,
        ):
            raise ValueError("action_sequences must have shape [B,50,2]")
        action_sequences = torch.clamp(
            action_sequences, self._action_min, self._action_max
        )
        trajectories = self.rollout_backend.evaluate_context_batch(
            history, initial_state, current_action, action_sequences
        ).to(self.device)
        components = self.context_batch_cost_components(
            trajectories, action_sequences, reference, current_action
        )
        cost = sum(components.values())
        return {
            "action_sequences": action_sequences.detach(),
            "trajectories": trajectories.detach(),
            "cost": cost.detach(),
            "cost_components": {
                name: value.detach() for name, value in components.items()
            },
        }

    @torch.no_grad()
    def hard_guard_action_sequence(
        self,
        initial_state,
        current_action,
        history,
        reference,
        warm_mppi_sequence,
        proposal_sequence,
        guard_state: Optional[Dict[str, int]] = None,
        switch_margin: float = 0.0,
        min_dwell: int = 0,
        warm_hard_return: bool = False,
    ) -> Dict[str, object]:
        """Choose the lower-model-cost sequence without blending actions.

        Candidate zero is always the original warm-MPPI weighted output and
        candidate one is the deterministic proposal.  Consequently the selected
        model cost cannot exceed the original controller output's evaluated cost
        when ``guard_state`` is None or the hysteresis parameters are zero.

        Optional hysteresis: ``guard_state`` carries ``selected_index`` and
        ``dwell`` across steps.  A branch switch is only allowed once the
        incumbent has been held for ``min_dwell`` steps and the challenger's
        model cost undercuts the incumbent by more than ``switch_margin``.
        With the defaults (0.0/0) and an initial warm incumbent the selection
        matches the plain argmin, including warm-wins-ties semantics.
        """
        candidates = torch.stack(
            (
                torch.as_tensor(warm_mppi_sequence, dtype=torch.float32, device=self.device),
                torch.as_tensor(proposal_sequence, dtype=torch.float32, device=self.device),
            )
        )
        result = self.evaluate_action_sequences(
            initial_state,
            current_action,
            history,
            reference,
            candidates,
        )
        if guard_state is None:
            selected_index = int(torch.argmin(result["cost"]).item())
        else:
            incumbent = int(guard_state.get("selected_index", 0))
            dwell = int(guard_state.get("dwell", 0))
            incumbent_cost = float(result["cost"][incumbent].item())
            challenger = 1 - incumbent
            challenger_cost = float(result["cost"][challenger].item())
            # Asymmetric warm return: leaving the actor branch back to warm
            # bypasses margin/dwell so the constructive warm hard floor is
            # preserved step-by-step; hysteresis only smooths actor entry.
            margin = (
                0.0 if (warm_hard_return and challenger == 0) else switch_margin
            )
            dwell_gate = 0 if (warm_hard_return and challenger == 0) else min_dwell
            if dwell >= dwell_gate and challenger_cost < incumbent_cost - margin:
                selected_index = challenger
                dwell = 0
            else:
                selected_index = incumbent
                dwell += 1
            guard_state["selected_index"] = selected_index
            guard_state["dwell"] = dwell
        selected_sequence = result["action_sequences"][selected_index]
        result.update(
            {
                "selected_index": selected_index,
                "selected_name": "warm_mppi" if selected_index == 0 else "proposal",
                "selected_action_sequence": selected_sequence.detach(),
                "selected_action": selected_sequence[0].detach(),
                "selected_cost": result["cost"][selected_index].detach(),
                "warm_cost": result["cost"][0].detach(),
                "proposal_cost": result["cost"][1].detach(),
            }
        )
        return result

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
            noise = self._sample_knot_noise()
            sampling_mean_knots = mean_knots
            raw_sampled_knots = sampling_mean_knots[None, :, :] + noise
            sampled_knots = raw_sampled_knots
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
                noise,
                sampling_mean_knots,
                raw_sampled_knots,
                sampled_knots,
            )

        assert last is not None
        (
            sampled_action,
            trajectory,
            cost,
            cost_components,
            weight,
            weighted_sequence,
            noise,
            sampling_mean_knots,
            raw_sampled_knots,
            sampled_knots,
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
            "sampling_noise_knots": noise.detach(),
            "sampling_mean_knots": sampling_mean_knots.detach(),
            "raw_sampled_knots": raw_sampled_knots.detach(),
            "sampled_knots": sampled_knots.detach(),
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
            "sampling_mode": self.params.sampling_mode,
        }
        full_trajectory = getattr(
            self.rollout_backend, "last_full_trajectory", None
        )
        if full_trajectory is not None:
            info["sampled_trajectories_full"] = full_trajectory.detach()
        return action.detach(), new_state, info
