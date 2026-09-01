#!/usr/bin/env python3
"""Independent DBM replay validation for the direct task-loss Actor pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs


def replay(actions, indices, states, current, reference, backend, weights,
           params, device):
    result = []
    for begin in range(0, len(indices), 128):
        local = indices[begin:begin + 128]
        knots = torch.from_numpy(actions[begin:begin + 128]).to(device)
        with torch.no_grad():
            cost = batched_cost(
                backend, weights,
                interpolate_knots(knots, params.horizon).unsqueeze(1),
                torch.from_numpy(states[local]).to(device),
                torch.from_numpy(current[local]).to(device),
                torch.from_numpy(reference[local]).to(device),
            )[:, 0]
        result.append(cost.cpu().numpy())
    return np.concatenate(result)


def distribution_errors(actual, expected):
    values = {
        "count": len(actual), "mean": np.mean(actual),
        "p05": np.quantile(actual, 0.05), "median": np.median(actual),
        "p95": np.quantile(actual, 0.95), "minimum": np.min(actual),
        "maximum": np.max(actual),
    }
    return max(abs(float(values[key]) - float(expected[key])) for key in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--bank-root", type=Path,
        default=Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"),
    )
    parser.add_argument(
        "--gt-v1", type=Path,
        default=Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    summary = json.loads((args.artifact / "summary.json").read_text())
    if summary["contract"]["critic_used"]:
        raise AssertionError("task-loss contract unexpectedly used a Critic")
    if summary["contract"]["formal_validation_loaded"] or summary["contract"]["test_loaded"]:
        raise AssertionError("sealed split contract violated")
    with np.load(args.bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, args.gt_v1
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    device = torch.device(args.device)
    errors = {}
    for size, record in summary["tiny"].items():
        payload = torch.load(args.artifact / f"tiny_{size}.pt", map_location="cpu")
        indices = np.asarray(payload["subset_indices"], np.int64)
        action = np.asarray(payload["final_action"], np.float32)
        cost = replay(action, indices, states, current, reference, backend, weights, params, device)
        errors[f"tiny_{size}_final_distribution"] = distribution_errors(
            cost, record["metrics"]["final_cost"]
        )
    if summary["full"] is not None:
        payload = torch.load(args.artifact / "full_train_fold0.pt", map_location="cpu")
        for name in ("train", "heldout"):
            indices = np.asarray(payload[f"{name}_indices"], np.int64)
            action = np.asarray(payload[f"{name}_action"], np.float32)
            cost = replay(action, indices, states, current, reference, backend, weights, params, device)
            expected = summary["full"]["train" if name == "train" else "episode_heldout"]["final_cost"]
            errors[f"full_{name}_final_distribution"] = distribution_errors(cost, expected)
    passed = max(errors.values()) <= 1e-4
    report = {
        "qualification": (
            "DBM_TASK_LOSS_ACTOR_VALIDATION_PASS" if passed
            else "DBM_TASK_LOSS_ACTOR_VALIDATION_FAIL"
        ),
        "maximum_summary_metric_error": max(errors.values()),
        "checks": errors,
        "critic_used": False,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.artifact / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report["qualification"], json.dumps(errors))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
