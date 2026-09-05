import math

import torch
from torch import nn

from car_foundation.query_deployment import QueryDeploymentModel, QueryHistoryBuffer


class _IdentityPosition(nn.Module):
    def forward(self, value):
        return value


class _FakeQueryCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_encoding = nn.ModuleDict(
            {"history": _IdentityPosition(), "action": _IdentityPosition()}
        )
        self.embedding = nn.ModuleDict({"output": nn.Identity()})
        self.tgt_mask = None

    def _build_history_emb(self, history):
        return history[..., :4]

    def _build_action_emb(
        self, history, future_action, context, nominal_state, nominal_transition
    ):
        return (
            nominal_transition
            + context[:, None]
            + history[..., :4].mean(dim=1, keepdim=True)
        )

    def transformer_decoder(self, tgt, memory, tgt_mask=None):
        del tgt_mask
        return tgt + memory.mean(dim=1, keepdim=True)


def _fake_deployment_model() -> QueryDeploymentModel:
    model = QueryDeploymentModel.__new__(QueryDeploymentModel)
    nn.Module.__init__(model)
    model.model = _FakeQueryCore()
    for name, size, fill in (
        ("history_mean", 5, 0.0),
        ("history_std", 5, 1.0),
        ("context_mean", 4, 0.0),
        ("context_std", 4, 1.0),
        ("nominal_state_mean", 5, 0.0),
        ("nominal_state_std", 5, 1.0),
        ("nominal_transition_mean", 4, 0.0),
        ("nominal_transition_std", 4, 1.0),
        ("residual_mean", 4, 0.0),
        ("residual_std", 4, 1.0),
    ):
        model.register_buffer(name, torch.full((size,), fill))
    model.dt = 0.05
    model.wheelbase = 2.8
    model.steering_ratio = 14.0
    model.steering_offset = 0.0
    return model


def test_history_buffer_matches_dataset_body_frame_transition_protocol():
    buffer = QueryHistoryBuffer(history_length=2, dt=0.05)
    previous = torch.tensor([1.0, 2.0, math.pi / 2, 4.0, 0.1])
    current = torch.tensor([1.0, 3.0, math.pi / 2 + 0.005, 4.2, 0.12])
    buffer.append(previous, torch.tensor([0.0, 0.0]))
    buffer.append(current, torch.tensor([0.3, -0.2]))

    assert not buffer.ready
    try:
        buffer.tensor()
    except RuntimeError as error:
        assert "1/2" in str(error)
    else:
        raise AssertionError("incomplete history must raise RuntimeError")

    following = torch.tensor([0.0, 3.0, math.pi / 2 + 0.010, 4.3, 0.11])
    buffer.append(following, torch.tensor([0.4, -0.1]))
    history = buffer.tensor()
    assert history.shape == (1, 2, 7)
    torch.testing.assert_close(
        history[0, 0],
        torch.tensor([1.0, 0.0, 0.005, 0.2, 0.02, 0.3, -0.2]),
        atol=1e-6,
        rtol=0,
    )


def test_constant_motion_prime_produces_model_shaped_history():
    buffer = QueryHistoryBuffer(dt=0.05)
    buffer.prime_constant_motion(
        torch.tensor([0.0, 0.0, 0.2, 10.0, 0.1]),
        torch.tensor([0.3, -0.2]),
    )
    history = buffer.tensor()
    assert history.shape == (1, 250, 7)
    torch.testing.assert_close(
        history[0, 0],
        torch.tensor([0.5, 0.0, 0.005, 0.0, 0.0, 0.3, -0.2]),
    )


def test_context_batch_matches_independent_single_context_rollouts():
    torch.manual_seed(7)
    model = _fake_deployment_model().eval()
    batch_size = 4
    history = torch.randn(batch_size, 250, 7) * 0.01
    state = torch.randn(batch_size, 5) * 0.01
    state[:, 3] += 4.0
    current_action = torch.randn(batch_size, 2) * 0.05
    future_action = torch.randn(batch_size, 50, 2) * 0.05

    batched = model.forward_context_batch(
        history, state, current_action, future_action
    )
    sequential = torch.cat([
        model(
            history[index:index + 1],
            state[index:index + 1],
            current_action[index:index + 1],
            future_action[index:index + 1],
        )
        for index in range(batch_size)
    ])
    torch.testing.assert_close(batched, sequential, atol=1e-6, rtol=1e-6)


def test_context_batch_rejects_misaligned_inputs():
    model = _fake_deployment_model().eval()
    history = torch.zeros(2, 250, 7)
    state = torch.zeros(2, 5)
    current_action = torch.zeros(2, 2)
    future_action = torch.zeros(2, 50, 2)
    try:
        model.forward_context_batch(history, state[:1], current_action, future_action)
    except ValueError as error:
        assert "initial_state" in str(error)
    else:
        raise AssertionError("misaligned context batches must be rejected")
