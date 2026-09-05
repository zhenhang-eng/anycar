#!/usr/bin/env python3
"""Batched direct Query costs for aligned physical contexts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.mppi import TorchMPPIController


def interpolate_knots(knots: np.ndarray, horizon: int = 50) -> np.ndarray:
    """Match the established CPU float32 knot interpolation exactly."""
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    return (
        F.interpolate(
            tensor.transpose(1, 2), size=horizon, mode="linear", align_corners=True
        )
        .transpose(1, 2)
        .numpy()
    )


def _check_cost_weights(
    controller: TorchMPPIController, weights: dict[str, float]
) -> None:
    for name, value in weights.items():
        if not hasattr(controller.cost_weights, name):
            raise ValueError(f"unknown Query cost weight {name!r}")
        configured = float(getattr(controller.cost_weights, name))
        if configured != float(value):
            raise ValueError(
                f"controller cost weight {name}={configured} does not match {value}"
            )


def batched_direct_cost(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    knots: np.ndarray,
    weights: dict[str, float],
    *,
    context_batch_size: int = 120,
) -> np.ndarray:
    """Evaluate one knot sequence per state using aligned context batches.

    The mathematical rollout and cost are unchanged.  Floating-point reduction
    order inside GPU kernels can differ from repeated batch-size-one calls, so
    callers that require historical bitwise replay should keep using the legacy
    sequential helper.  Qualification code should verify that checkpoint choices
    and registered metric tolerances are unchanged before promoting a batch size.
    """
    rows = np.asarray(rows, np.int64).reshape(-1)
    knots = np.asarray(knots, np.float32)
    if knots.shape != (len(rows), controller.params.num_knots, controller.params.action_dim):
        raise ValueError("knots must have shape [len(rows),8,2]")
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be positive")
    if len(rows) == 0:
        return np.empty(0, np.float32)
    _check_cost_weights(controller, weights)
    actions = interpolate_knots(knots, controller.params.horizon)
    costs = []
    for start in range(0, len(rows), context_batch_size):
        local = rows[start:start + context_batch_size]
        result: dict[str, Any] = controller.evaluate_context_action_sequences(
            data["state"][local],
            data["current_action"][local],
            data["history"][local],
            data["reference"][local],
            actions[start:start + len(local)],
        )
        costs.append(result["cost"].cpu().numpy().astype(np.float32))
    return np.concatenate(costs)
