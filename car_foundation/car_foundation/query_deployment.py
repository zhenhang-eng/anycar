"""Deployment graph and runtime backends for the deterministic Query model.

The public rollout protocol intentionally matches the current training data:

* raw transition history: ``[1, 250, 7]``;
* current observable state: ``[1, 5]`` as ``[x, y, yaw, vx, yawrate]``;
* current action: ``[1, 2]`` as ``[acceleration, steering_command]``;
* candidate future actions: ``[N, 50, 2]``.

History is encoded once and shared by all MPPI candidates.  Both the PyTorch
and ONNX Runtime backends execute the same full graph, including input
normalization, nominal kinematics, residual de-normalization, and consistent
state rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union

import numpy as np
import torch
from torch import nn

from car_foundation.models import TorchTransformerDecoderKinematicQueryMLP


PathLike = Union[str, Path]


def _as_float_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().clone()


def build_query_model(checkpoint: dict, device: torch.device) -> nn.Module:
    """Construct the exact deterministic Query architecture in a checkpoint."""
    if checkpoint.get("variant") != "query":
        raise ValueError(
            "Expected a deterministic Query checkpoint, got "
            f"variant={checkpoint.get('variant')!r}"
        )
    args = checkpoint["args"]
    model = TorchTransformerDecoderKinematicQueryMLP(
        state_dim=5,
        action_dim=2,
        output_dim=4,
        latent_dim=args["latent_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        device=device,
        dropout=args["dropout"],
        history_length=args["history_length"],
        prediction_length=args["prediction_length"],
        compressed_history_length=42,
        current_dim=4,
        fusion_hidden_dim=args["fusion_hidden_dim"],
        nominal_state_dim=5,
        nominal_transition_dim=4,
        query_hidden_dim=args["query_hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


class QueryDeploymentModel(nn.Module):
    """Complete deterministic Query rollout with an MPPI-oriented interface."""

    history_length = 250
    horizon = 50
    state_dim = 5
    action_dim = 2

    def __init__(self, model: nn.Module, checkpoint: dict):
        super().__init__()
        self.model = model
        args = checkpoint["args"]
        if args["history_length"] != self.history_length:
            raise ValueError(
                f"Expected history_length={self.history_length}, "
                f"got {args['history_length']}"
            )
        if args["prediction_length"] != self.horizon:
            raise ValueError(
                f"Expected prediction_length={self.horizon}, "
                f"got {args['prediction_length']}"
            )

        stats = checkpoint["stats"]
        for name in (
            "history",
            "context",
            "residual",
            "nominal_state",
            "nominal_transition",
        ):
            mean, std = stats[name]
            self.register_buffer(f"{name}_mean", _as_float_tensor(mean))
            self.register_buffer(f"{name}_std", _as_float_tensor(std))

        params = checkpoint["params"]
        self.dt = float(params["dt"])
        self.wheelbase = float(params["wheelbase"])
        self.steering_ratio = float(params["steering_ratio"])
        self.steering_offset = float(params["steering_offset"])

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: PathLike, device: Union[str, torch.device] = "cuda"
    ) -> "QueryDeploymentModel":
        device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        return cls(build_query_model(checkpoint, device), checkpoint).to(device).eval()

    def _validate_inputs(self, history, initial_state, current_action, future_action):
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [1, 250, 7]")
        if history.shape[0] != 1:
            raise ValueError("MPPI deployment history batch must be exactly one")
        if tuple(initial_state.shape) != (1, self.state_dim):
            raise ValueError("initial_state must have shape [1, 5]")
        if tuple(current_action.shape) != (1, self.action_dim):
            raise ValueError("current_action must have shape [1, 2]")
        if future_action.ndim != 3 or tuple(future_action.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("future_action must have shape [N, 50, 2]")

    def _kinematic_transition(self, state, action):
        vx = state[..., 3]
        yawrate = state[..., 4]
        acceleration = action[..., 0]
        front_wheel_angle = (
            action[..., 1] - self.steering_offset
        ) / self.steering_ratio
        dvx = acceleration * self.dt
        vx_mid = vx + 0.5 * dvx
        yawrate_next = vx_mid * torch.tan(front_wheel_angle) / self.wheelbase
        return torch.stack(
            (
                vx_mid * self.dt,
                torch.zeros_like(vx),
                dvx,
                yawrate_next - yawrate,
            ),
            dim=-1,
        )

    def _apply_transition(self, state, transition):
        yaw = state[..., 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        next_yaw = yaw + state[..., 4] * self.dt
        return torch.stack(
            (
                state[..., 0]
                + transition[..., 0] * cos_yaw
                - transition[..., 1] * sin_yaw,
                state[..., 1]
                + transition[..., 0] * sin_yaw
                + transition[..., 1] * cos_yaw,
                torch.atan2(torch.sin(next_yaw), torch.cos(next_yaw)),
                state[..., 3] + transition[..., 2],
                state[..., 4] + transition[..., 3],
            ),
            dim=-1,
        )

    def _nominal_rollout(self, initial_state, action):
        state = initial_state
        states = []
        transitions = []
        for step in range(self.horizon):
            transition = self._kinematic_transition(state, action[:, step])
            state = self._apply_transition(state, transition)
            transitions.append(transition)
            states.append(state)
        return torch.stack(states, dim=1), torch.stack(transitions, dim=1)

    def _relative_state(self, initial_state, states):
        initial = initial_state[:, None, :]
        dx_world = states[..., 0] - initial[..., 0]
        dy_world = states[..., 1] - initial[..., 1]
        initial_yaw = initial[..., 2]
        cos_yaw = torch.cos(initial_yaw)
        sin_yaw = torch.sin(initial_yaw)
        return torch.stack(
            (
                dx_world * cos_yaw + dy_world * sin_yaw,
                -dx_world * sin_yaw + dy_world * cos_yaw,
                torch.atan2(
                    torch.sin(states[..., 2] - initial_yaw),
                    torch.cos(states[..., 2] - initial_yaw),
                ),
                states[..., 3],
                states[..., 4],
            ),
            dim=-1,
        )

    def _residual_rollout(self, initial_state, action, residual):
        state = initial_state
        states = []
        for step in range(self.horizon):
            transition = self._kinematic_transition(state, action[:, step])
            state = self._apply_transition(state, transition + residual[:, step])
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(self, history, initial_state, current_action, future_action):
        self._validate_inputs(history, initial_state, current_action, future_action)
        candidate_count = future_action.shape[0]

        normalized_history_state = (
            history[..., :5] - self.history_mean
        ) / self.history_std
        normalized_history = torch.cat(
            (normalized_history_state, history[..., 5:7]), dim=-1
        )

        # Encode the single real history once, then share it across candidates.
        history_memory = self.model._build_history_emb(normalized_history)
        history_memory = self.model.position_encoding["history"](history_memory)
        history_memory = history_memory.expand(candidate_count, -1, -1)

        initial = initial_state.expand(candidate_count, -1)
        # Match NuPlanKinematicResidualDataset exactly.  At the first future
        # transition, context throttle and future_action[:, 0, 0] share the
        # same raw index, while context steer is the current (unshifted) steer.
        context = torch.cat(
            (
                initial[:, 3:5],
                future_action[:, 0, 0:1],
                current_action[:, 1:2].expand(candidate_count, -1),
            ),
            dim=-1,
        )
        context = (context - self.context_mean) / self.context_std

        nominal_absolute, nominal_transition = self._nominal_rollout(
            initial, future_action
        )
        nominal_state = self._relative_state(initial, nominal_absolute)
        nominal_state = (
            nominal_state - self.nominal_state_mean
        ) / self.nominal_state_std
        nominal_transition_normalized = (
            nominal_transition - self.nominal_transition_mean
        ) / self.nominal_transition_std

        # current_state is explicit, so _build_action_emb does not read history.
        history_for_query = normalized_history.expand(candidate_count, -1, -1)
        action_embedding = self.model._build_action_emb(
            history_for_query,
            future_action,
            context,
            nominal_state,
            nominal_transition_normalized,
        )
        action_embedding = self.model.position_encoding["action"](action_embedding)
        hidden = self.model.transformer_decoder(
            tgt=action_embedding,
            memory=history_memory,
            tgt_mask=self.model.tgt_mask,
        )
        residual_normalized = self.model.embedding["output"](hidden)
        residual = residual_normalized * self.residual_std + self.residual_mean
        return self._residual_rollout(initial, future_action, residual)


class QueryRolloutBackend(Protocol):
    def __call__(
        self,
        history: torch.Tensor,
        initial_state: torch.Tensor,
        current_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor: ...


class TorchQueryRolloutBackend:
    """Thin inference-mode adapter around :class:`QueryDeploymentModel`."""

    def __init__(self, model: QueryDeploymentModel):
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def __call__(self, history, initial_state, current_action, future_action):
        with torch.inference_mode():
            return self.model(
                history.to(self.device),
                initial_state.to(self.device),
                current_action.to(self.device),
                future_action.to(self.device),
            )


class OnnxQueryRolloutBackend:
    """ONNX Runtime implementation of the exact Query rollout protocol."""

    def __init__(
        self,
        onnx_path: PathLike,
        provider: str = "cuda",
        output_device: Union[str, torch.device] = "cpu",
    ):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime is required for the ONNX backend") from error

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(Path(onnx_path)), sess_options=options, providers=providers
        )
        if provider == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "CUDAExecutionProvider was requested but not activated: "
                f"{self.session.get_providers()}"
            )
        self.output_device = torch.device(output_device)

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)

    def __call__(self, history, initial_state, current_action, future_action):
        trajectory = self.session.run(
            None,
            {
                "history": self._numpy(history),
                "initial_state": self._numpy(initial_state),
                "current_action": self._numpy(current_action),
                "future_action": self._numpy(future_action),
            },
        )[0]
        return torch.from_numpy(trajectory).to(self.output_device)


def export_query_onnx(
    model: QueryDeploymentModel,
    output_path: PathLike,
    opset: int = 17,
    candidate_batch: int = 2,
) -> Path:
    """Export a dynamic-candidate ONNX graph and run the ONNX checker."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required to export the Query model") from error

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    history = torch.zeros(1, model.history_length, 7, device=device)
    initial_state = torch.zeros(1, model.state_dim, device=device)
    current_action = torch.zeros(1, model.action_dim, device=device)
    future_action = torch.zeros(
        candidate_batch, model.horizon, model.action_dim, device=device
    )
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (history, initial_state, current_action, future_action),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=(
                "history",
                "initial_state",
                "current_action",
                "future_action",
            ),
            output_names=("trajectory",),
            dynamic_axes={
                "future_action": {0: "candidate_batch"},
                "trajectory": {0: "candidate_batch"},
            },
        )
    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


@dataclass
class QueryHistoryBuffer:
    """Build the model's 250 transition-history tokens from online states."""

    history_length: int = 250
    dt: float = 0.05

    def __post_init__(self):
        self._tokens = []
        self._previous_state = None

    @property
    def ready(self) -> bool:
        return len(self._tokens) == self.history_length

    def clear(self):
        self._tokens.clear()
        self._previous_state = None

    def prime_constant_motion(self, state, action):
        """Warm-start with a physically plausible constant-motion history."""
        state = _as_float_tensor(state).reshape(5)
        action = _as_float_tensor(action).reshape(2)
        token = torch.stack(
            (
                state[3] * self.dt,
                torch.zeros_like(state[3]),
                state[4] * self.dt,
                torch.zeros_like(state[3]),
                torch.zeros_like(state[4]),
                action[0],
                action[1],
            )
        )
        self._tokens = [token.clone() for _ in range(self.history_length)]
        self._previous_state = state.clone()

    def append(self, state, action):
        state = _as_float_tensor(state).reshape(5)
        action = _as_float_tensor(action).reshape(2)
        if self._previous_state is None:
            self._previous_state = state.clone()
            return

        previous = self._previous_state
        dx_world = state[0] - previous[0]
        dy_world = state[1] - previous[1]
        cos_yaw = torch.cos(previous[2])
        sin_yaw = torch.sin(previous[2])
        dx_body = dx_world * cos_yaw + dy_world * sin_yaw
        dy_body = -dx_world * sin_yaw + dy_world * cos_yaw
        dyaw_raw = state[2] - previous[2]
        dyaw = torch.atan2(torch.sin(dyaw_raw), torch.cos(dyaw_raw))
        token = torch.stack(
            (
                dx_body,
                dy_body,
                dyaw,
                state[3] - previous[3],
                state[4] - previous[4],
                action[0],
                action[1],
            )
        )
        self._tokens.append(token)
        if len(self._tokens) > self.history_length:
            self._tokens.pop(0)
        self._previous_state = state.clone()

    def tensor(self, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        if not self.ready:
            raise RuntimeError(
                f"Query history is not ready: {len(self._tokens)}/{self.history_length}"
            )
        return torch.stack(self._tokens, dim=0).unsqueeze(0).to(device)
