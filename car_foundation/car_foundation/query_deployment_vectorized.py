"""Experimental standard-ONNX prefix formulation; NOT a deployment default.

Same weights, input contract and real-arithmetic dynamics as QueryDeploymentModel.
Floating-point reassociation requires differential trajectory/cost qualification.
No custom operators, plugins, CUDA Graph, or reduced precision are required.
"""
import torch

from .query_deployment import QueryDeploymentModel


class PrefixQueryDeploymentModel(QueryDeploymentModel):
    def __init__(self, model, checkpoint):
        super().__init__(model, checkpoint)
        # Row-vector times upper triangular ones = inclusive prefix sum.
        self.register_buffer("prefix_matrix", torch.triu(torch.ones(self.horizon, self.horizon)))

    def _prefix(self, values):
        return values @ self.prefix_matrix

    @staticmethod
    def _previous(initial, values):
        return torch.cat((initial, values[:, :-1]), dim=1)

    def _parallel_rollout(self, initial, action, residual=None):
        dv = action[..., 0] * self.dt
        delta_v = dv if residual is None else dv + residual[..., 2]
        velocity = initial[:, 3:4] + self._prefix(delta_v)
        previous_v = self._previous(initial[:, 3:4], velocity)
        mid_v = previous_v + .5 * dv
        front = (action[..., 1] - self.steering_offset) / self.steering_ratio
        nominal_rate = mid_v * torch.tan(front) / self.wheelbase
        rate = nominal_rate if residual is None else nominal_rate + residual[..., 3]
        previous_rate = self._previous(initial[:, 4:5], rate)
        yaw_unwrapped = initial[:, 2:3] + self._prefix(previous_rate * self.dt)
        yaw = torch.atan2(torch.sin(yaw_unwrapped), torch.cos(yaw_unwrapped))
        previous_yaw = self._previous(initial[:, 2:3], yaw)
        dx = mid_v * self.dt
        dy = torch.zeros_like(dx)
        if residual is not None:
            dx = dx + residual[..., 0]
            dy = residual[..., 1]
        co, si = torch.cos(previous_yaw), torch.sin(previous_yaw)
        x = initial[:, 0:1] + self._prefix(dx * co - dy * si)
        y = initial[:, 1:2] + self._prefix(dx * si + dy * co)
        states = torch.stack((x, y, yaw, velocity, rate), dim=-1)
        # The transformer consumes the ORIGINAL nominal transition definition.
        transitions = torch.stack((mid_v * self.dt, torch.zeros_like(dv),
                                   dv, nominal_rate - previous_rate), dim=-1)
        return states, transitions

    def _nominal_rollout(self, initial_state, action):
        return self._parallel_rollout(initial_state, action)

    def _residual_rollout(self, initial_state, action, residual):
        return self._parallel_rollout(initial_state, action, residual)[0]
