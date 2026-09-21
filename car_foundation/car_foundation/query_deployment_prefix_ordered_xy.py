"""Conservative prefix experiment: retain original FP32 x/y accumulation order."""
import torch
from .query_deployment_vectorized import PrefixQueryDeploymentModel


class OrderedXYPrefixQueryDeploymentModel(PrefixQueryDeploymentModel):
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
        # Parallel products, original sequential left-associative additions.
        # Do not combine dx*cos-dy*sin before adding the previous coordinate.
        xc, ys, xs, yc = dx * co, dy * si, dx * si, dy * co
        x, y = initial[:, 0], initial[:, 1]
        xx, yy = [], []
        for t in range(self.horizon):
            x = x + xc[:, t] - ys[:, t]
            y = y + xs[:, t] + yc[:, t]
            xx.append(x); yy.append(y)
        states = torch.stack((torch.stack(xx,1),torch.stack(yy,1),yaw,velocity,rate),-1)
        transitions = torch.stack((mid_v * self.dt, torch.zeros_like(dv),
                                   dv, nominal_rate - previous_rate), dim=-1)
        return states, transitions
