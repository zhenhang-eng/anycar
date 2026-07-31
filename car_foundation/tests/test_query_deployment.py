import math

import torch

from car_foundation.query_deployment import QueryHistoryBuffer


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
