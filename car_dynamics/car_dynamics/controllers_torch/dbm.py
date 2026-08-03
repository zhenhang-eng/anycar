"""Batched PyTorch dynamic-bicycle rollout for MPPI comparisons.

The public interface follows the current Query deployment protocol.  The
analytic model internally propagates the DBM's lateral velocity, while the
observable five-state input initializes lateral velocity to zero because it is
not part of the current model state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchDBMParams:
    """Numeric Quick Start vehicle parameters at the Query model time step."""

    lf: float = 0.1008
    lr: float = 0.1092
    mass: float = 4.0
    dt: float = 0.05
    integration_substeps: int = 1
    inertia_z: float = 0.07
    throttle_scale: float = 8.0
    throttle_velocity_gain: float = 0.0
    steering_scale: float = 0.36
    steering_bias: float = 0.025
    friction: float = 0.8
    pacejka_c_front: float = 1.0
    pacejka_c_rear: float = 1.0
    pacejka_b_front: float = 20.0
    pacejka_b_rear: float = 20.0
    center_of_mass_height: float = 0.1
    rolling_friction: float = 0.05

    def __post_init__(self):
        if self.dt != 0.05:
            raise ValueError("DBM comparison must use the Query protocol dt=0.05")
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be positive")


class TorchDynamicBicycleRolloutBackend:
    """Roll out numeric-simulator DBM parameters with torch RK4 integration."""

    horizon = 50
    state_dim = 5
    action_dim = 2
    gravity = 9.81

    def __init__(self, params: TorchDBMParams | None = None):
        self.params = params or TorchDBMParams()
        self.initial_lateral_velocity = 0.0
        self.last_full_trajectory: torch.Tensor | None = None

    def set_initial_lateral_velocity(self, lateral_velocity: float) -> None:
        """Set the simulator-observed ``vy`` used to initialize each rollout."""
        self.initial_lateral_velocity = float(lateral_velocity)

    def _derivative(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        p = self.params
        x, y, yaw, vx, vy, yawrate = state.unbind(dim=-1)
        del x, y
        acceleration_command, steering_command = action.unbind(dim=-1)
        steer = steering_command * p.steering_scale + p.steering_bias
        speed = torch.sqrt(vx.square() + vy.square())
        longitudinal_force = (
            acceleration_command * p.throttle_scale
            + p.throttle_velocity_gain * speed
            - p.rolling_friction * p.mass * self.gravity * torch.sign(vx)
        )

        denominator = torch.maximum(vx, torch.full_like(vx, 0.5))
        slip_front = steer - torch.atan((p.lf * yawrate + vy) / denominator)
        slip_rear = torch.atan((p.lr * yawrate - vy) / denominator)
        normal_front = (
            0.5 * p.mass * self.gravity * p.lr / (p.lf + p.lr)
            - 0.5 * p.center_of_mass_height * longitudinal_force / (p.lf + p.lr)
        )
        normal_rear = (
            0.5 * p.mass * self.gravity * p.lf / (p.lf + p.lr)
            + 0.5 * p.center_of_mass_height * longitudinal_force / (p.lf + p.lr)
        )
        lateral_front = p.friction * normal_front * torch.sin(
            p.pacejka_c_front * torch.atan(p.pacejka_b_front * slip_front)
        )
        lateral_rear = p.friction * normal_rear * torch.sin(
            p.pacejka_c_rear * torch.atan(p.pacejka_b_rear * slip_rear)
        )

        return torch.stack(
            (
                vx * torch.cos(yaw) - vy * torch.sin(yaw),
                vx * torch.sin(yaw) + vy * torch.cos(yaw),
                yawrate,
                (
                    longitudinal_force
                    - lateral_front * torch.sin(steer)
                    + vy * yawrate * p.mass
                )
                / p.mass,
                (
                    lateral_rear
                    + lateral_front * torch.cos(steer)
                    - vx * yawrate * p.mass
                )
                / p.mass,
                (
                    lateral_front * p.lf * torch.cos(steer)
                    - lateral_rear * p.lr
                )
                / p.inertia_z,
            ),
            dim=-1,
        )

    def _step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dt = self.params.dt / self.params.integration_substeps
        for _ in range(self.params.integration_substeps):
            k1 = self._derivative(state, action)
            k2 = self._derivative(state + 0.5 * dt * k1, action)
            k3 = self._derivative(state + 0.5 * dt * k2, action)
            k4 = self._derivative(state + dt * k3, action)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        state = state.clone()
        state[..., 2] = torch.atan2(torch.sin(state[..., 2]), torch.cos(state[..., 2]))
        return state

    @torch.no_grad()
    def step_full_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Advance batched ``[x,y,yaw,vx,vy,yawrate]`` states by 0.05 s."""
        if state.ndim != 2 or state.shape[1] != 6:
            raise ValueError("state must have shape [N, 6]")
        if action.shape != (state.shape[0], 2):
            raise ValueError("action must have shape [N, 2]")
        return self._step(state, action)

    @torch.no_grad()
    def rollout_full_state(
        self,
        history: torch.Tensor,
        initial_state: torch.Tensor,
        current_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out and retain all six DBM states, including lateral velocity."""
        del history, current_action
        if initial_state.shape != (1, self.state_dim):
            raise ValueError("initial_state must have shape [1, 5]")
        if future_action.ndim != 3 or tuple(future_action.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("future_action must have shape [N, 50, 2]")

        candidate_count = future_action.shape[0]
        initial = initial_state.expand(candidate_count, -1)
        initial_vy = torch.full_like(
            initial[..., 3:4], self.initial_lateral_velocity
        )
        state = torch.cat(
            (initial[..., :4], initial_vy, initial[..., 4:5]),
            dim=-1,
        )
        output = []
        for step in range(self.horizon):
            state = self.step_full_state(state, future_action[:, step])
            output.append(state)
        return torch.stack(output, dim=1)

    @torch.no_grad()
    def __call__(
        self,
        history: torch.Tensor,
        initial_state: torch.Tensor,
        current_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        full_trajectory = self.rollout_full_state(
            history, initial_state, current_action, future_action
        )
        self.last_full_trajectory = full_trajectory
        return full_trajectory[..., [0, 1, 2, 3, 5]]
