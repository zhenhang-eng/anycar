#!/usr/bin/env python3
"""Independently replay the train-only DBM output-support A/B artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_dbm_task_loss_actor import DEFAULT_BANK, DEFAULT_GT_V1, rollout_cost


DEFAULT_ARTIFACT = Path("outputs/mppi_proposal/dbm_task_loss_support_ab_20260825_v1")


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def max_metric_error(left: dict, right: dict) -> float:
    keys = ("mean", "p05", "median", "p95", "minimum", "maximum")
    if int(left["count"]) != int(right["count"]):
        return float("inf")
    return max(abs(float(left[key]) - float(right[key])) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    summary = json.loads((args.artifact / "summary.json").read_text())
    parameters = summary["parameters"]
    bank_root = Path(parameters.get("bank_root", DEFAULT_BANK))
    gt_v1 = Path(parameters.get("gt_v1", DEFAULT_GT_V1))
    with np.load(bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, gt_v1
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    device = torch.device(args.device)
    subset = np.asarray(summary["subset_indices"], dtype=np.int64)

    reports = {}
    initial_actions = {}
    maximum_error = 0.0
    for mode in ("box1", "box3", "full"):
        payload = torch.load(args.artifact / f"support_{mode}.pt", map_location="cpu")
        saved_subset = np.asarray(payload["subset_indices"], dtype=np.int64)
        if not np.array_equal(saved_subset, subset):
            raise AssertionError(f"{mode}: subset mismatch")
        initial_actions[mode] = np.asarray(payload["initial_action"], dtype=np.float32)
        final_action = np.asarray(payload["final_action"], dtype=np.float32)
        if final_action.shape != (len(subset), 8, 2):
            raise AssertionError(f"{mode}: unexpected final action shape {final_action.shape}")
        if np.max(np.abs(final_action)) > 1.000001:
            raise AssertionError(f"{mode}: physical action bound violated")

        packed = torch.zeros(len(states), 8, 2, dtype=torch.float32, device=device)
        packed[torch.from_numpy(subset).to(device)] = torch.from_numpy(final_action).to(device)
        replay = rollout_cost(
            backend, weights, params, packed, states, current, reference,
            subset, 128, device,
        ).cpu().numpy()
        replay_dist = distribution(replay)
        recorded_dist = summary["records"][mode]["metrics"]["final_cost"]
        metric_error = max_metric_error(replay_dist, recorded_dist)
        maximum_error = max(maximum_error, metric_error)
        reports[mode] = {
            "replayed_final_cost": replay_dist,
            "recorded_final_cost": recorded_dist,
            "max_distribution_metric_abs_error": metric_error,
        }

    paired_initial_error = max(
        float(np.max(np.abs(initial_actions[left] - initial_actions[right])))
        for left, right in (("box1", "box3"), ("box1", "full"))
    )
    contract_initial_error = float(summary["contract"]["paired_initial_action_max_abs_error"])
    initial_error_delta = abs(paired_initial_error - contract_initial_error)
    contract_sealed = (
        summary["contract"]["formal_validation_loaded"] is False
        and summary["contract"]["test_loaded"] is False
    )
    passed = maximum_error <= 1e-4 and initial_error_delta <= 1e-9 and contract_sealed
    report = {
        "qualification": (
            "DBM_TASK_LOSS_SUPPORT_AB_INDEPENDENT_VALIDATION_PASS"
            if passed else "DBM_TASK_LOSS_SUPPORT_AB_INDEPENDENT_VALIDATION_FAIL"
        ),
        "artifact_qualification": summary["qualification"],
        "maximum_distribution_metric_abs_error": maximum_error,
        "paired_initial_action_max_abs_error": paired_initial_error,
        "paired_initial_action_contract_delta": initial_error_delta,
        "formal_validation_and_test_sealed": contract_sealed,
        "records": reports,
    }
    (args.artifact / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
