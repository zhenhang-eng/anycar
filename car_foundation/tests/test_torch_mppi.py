import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPIParams,
)


class _DummyRollout:
    def __call__(self, history, initial_state, current_action, future_action):
        del history, current_action
        batch = future_action.shape[0]
        state = initial_state.expand(batch, -1).clone()
        outputs = []
        for step in range(future_action.shape[1]):
            state = state.clone()
            state[:, 0] += future_action[:, step, 0] * 0.05
            state[:, 2] += future_action[:, step, 1] * 0.01
            state[:, 3] += future_action[:, step, 0] * 0.05
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def test_torch_mppi_runs_and_preserves_query_shapes():
    params = TorchMPPIParams(num_samples=16, seed=7)
    controller = TorchMPPIController(
        _DummyRollout(), params=params, device="cpu"
    )
    history = torch.zeros(1, 250, 7)
    initial_state = torch.zeros(5)
    current_action = torch.zeros(2)
    reference = torch.zeros(50, 4)
    reference[:, 0] = torch.linspace(0.0, 1.0, 50)
    reference[:, 3] = 1.0

    action, running_state, info = controller(
        initial_state, current_action, history, reference
    )
    assert action.shape == (2,)
    assert running_state.mean_knots.shape == (8, 2)
    assert info["trajectory"].shape == (50, 5)
    assert info["sampled_action_sequences"].shape == (16, 50, 2)
    assert info["sampled_trajectories"].shape == (16, 50, 5)
    assert info["optimized_action_sequence"].shape == (50, 2)
    assert torch.isfinite(action).all()
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert float(info["effective_sample_size"]) >= 1.0
    component_sum = sum(info["cost_components"].values())
    torch.testing.assert_close(component_sum, info["cost"])


def test_torch_dbm_backend_follows_query_rollout_protocol():
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(integration_substeps=1)
    )
    history = torch.zeros(1, 250, 7)
    initial_state = torch.tensor([[0.2, -0.1, 0.3, 1.2, 0.15]])
    current_action = torch.tensor([[0.1, -0.2]])
    future_action = torch.zeros(4, 50, 2)
    future_action[:, 0] = torch.tensor([0.4, -0.3])

    trajectory = backend(
        history, initial_state, current_action, future_action
    )
    assert trajectory.shape == (4, 50, 5)
    assert torch.isfinite(trajectory).all()
    torch.testing.assert_close(
        trajectory[0, 0],
        torch.tensor(
            [0.2579678, -0.08357857, 0.2939425, 1.2107577, -0.33135685]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_torch_dbm_backend_uses_observed_lateral_velocity():
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(integration_substeps=1)
    )
    backend.set_initial_lateral_velocity(-0.08)
    initial_state = torch.tensor([[0.2, -0.1, 0.3, 1.2, 0.15]])
    future_action = torch.zeros(1, 50, 2)
    future_action[:, 0] = torch.tensor([0.4, -0.3])
    trajectory = backend(
        torch.zeros(1, 250, 7),
        initial_state,
        torch.zeros(1, 2),
        future_action,
    )
    expected_full_state = backend.step_full_state(
        torch.tensor([[0.2, -0.1, 0.3, 1.2, -0.08, 0.15]]),
        torch.tensor([[0.4, -0.3]]),
    )
    torch.testing.assert_close(
        trajectory[0, 0], expected_full_state[0, [0, 1, 2, 3, 5]]
    )
