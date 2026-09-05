import torch

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams


class _AlignedFakeBackend:
    @staticmethod
    def _rollout(initial_state, future_action):
        batch_size, horizon, _ = future_action.shape
        initial = initial_state.expand(batch_size, -1)
        x = initial[:, 0:1] + torch.cumsum(future_action[..., 0], dim=1) * 0.05
        y = initial[:, 1:2].expand(-1, horizon)
        yaw = initial[:, 2:3] + torch.cumsum(future_action[..., 1], dim=1) * 0.01
        vx = initial[:, 3:4] + torch.cumsum(future_action[..., 0], dim=1) * 0.02
        yawrate = future_action[..., 1] * 0.1
        return torch.stack((x, y, yaw, vx, yawrate), dim=-1)

    def __call__(self, history, initial_state, current_action, future_action):
        del history, current_action
        return self._rollout(initial_state, future_action)

    def evaluate_context_batch(
        self, history, initial_state, current_action, future_action
    ):
        del history, current_action
        return self._rollout(initial_state, future_action)


def test_context_batch_cost_matches_independent_single_context_costs():
    torch.manual_seed(11)
    controller = TorchMPPIController(
        _AlignedFakeBackend(), TorchMPPIParams(), device="cpu"
    )
    batch_size = 5
    state = torch.randn(batch_size, 5) * 0.1
    state[:, 3] += 4.0
    current_action = torch.randn(batch_size, 2) * 0.05
    history = torch.randn(batch_size, 250, 7) * 0.01
    reference = torch.randn(batch_size, 51, 4) * 0.1
    action = torch.randn(batch_size, 50, 2) * 0.1

    batched = controller.evaluate_context_action_sequences(
        state, current_action, history, reference, action
    )
    sequential_costs = []
    for index in range(batch_size):
        result = controller.evaluate_action_sequences(
            state[index],
            current_action[index],
            history[index:index + 1],
            reference[index],
            action[index:index + 1],
        )
        sequential_costs.append(result["cost"])
    sequential = torch.cat(sequential_costs)
    torch.testing.assert_close(batched["cost"], sequential, atol=0, rtol=0)


def test_context_batch_requires_backend_support():
    controller = TorchMPPIController(
        lambda history, state, current_action, future_action: torch.zeros(
            len(future_action), 50, 5
        ),
        TorchMPPIParams(),
        device="cpu",
    )
    try:
        controller.evaluate_context_action_sequences(
            torch.zeros(1, 5),
            torch.zeros(1, 2),
            torch.zeros(1, 250, 7),
            torch.zeros(1, 51, 4),
            torch.zeros(1, 50, 2),
        )
    except TypeError as error:
        assert "aligned context batches" in str(error)
    else:
        raise AssertionError("unsupported rollout backends must be rejected")
