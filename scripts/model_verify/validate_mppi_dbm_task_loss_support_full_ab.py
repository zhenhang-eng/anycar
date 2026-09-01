#!/usr/bin/env python3
"""Independent DBM replay for the full train/heldout output-support A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_dbm_task_loss_actor import DEFAULT_BANK, DEFAULT_GT_V1, result_metrics, rollout_cost


DEFAULT_ARTIFACT = Path("outputs/mppi_proposal/dbm_task_loss_support_full_ab_20260825_v1")


def numeric_max_error(left, right) -> float:
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max((numeric_max_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def replay_actions(actions: np.ndarray, indices: np.ndarray, *, backend, weights,
                   params, states, current, reference, device) -> np.ndarray:
    packed = torch.zeros(len(states), 8, 2, dtype=torch.float32, device=device)
    packed[torch.from_numpy(indices).to(device)] = torch.from_numpy(actions).to(device)
    return rollout_cost(
        backend, weights, params, packed, states, current, reference,
        indices, 128, device,
    ).cpu().numpy()


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
    train_indices = np.asarray(summary["train_indices"], dtype=np.int64)
    heldout_indices = np.asarray(summary["heldout_indices"], dtype=np.int64)
    best_index = np.argmin(data["costs"][:, 16:24], axis=1)
    j16_cost = data["costs"][:, 16:24][np.arange(len(data["costs"])), best_index]

    reports = {}
    initial_all = {}
    maximum_error = 0.0
    replayed_metrics = {}
    for mode in ("box1", "box3"):
        payload = torch.load(args.artifact / f"support_{mode}.pt", map_location="cpu")
        if not np.array_equal(np.asarray(payload["train_indices"]), train_indices):
            raise AssertionError(f"{mode}: train split mismatch")
        if not np.array_equal(np.asarray(payload["heldout_indices"]), heldout_indices):
            raise AssertionError(f"{mode}: heldout split mismatch")
        initial_all[mode] = np.concatenate([
            np.asarray(payload["initial_train_action"], dtype=np.float32),
            np.asarray(payload["initial_heldout_action"], dtype=np.float32),
        ])
        replayed_metrics[mode] = {}
        reports[mode] = {}
        for split, indices in (("train", train_indices), ("episode_heldout", heldout_indices)):
            prefix = "train" if split == "train" else "heldout"
            initial_action = np.asarray(payload[f"initial_{prefix}_action"], dtype=np.float32)
            final_action = np.asarray(payload[f"{prefix}_action"], dtype=np.float32)
            if initial_action.shape != final_action.shape or final_action.shape != (len(indices), 8, 2):
                raise AssertionError(f"{mode}/{split}: action shape mismatch")
            if float(np.max(np.abs(final_action))) > 1.000001:
                raise AssertionError(f"{mode}/{split}: physical action bound violated")
            initial_cost = replay_actions(
                initial_action, indices, backend=backend, weights=weights, params=params,
                states=states, current=current, reference=reference, device=device,
            )
            final_cost = replay_actions(
                final_action, indices, backend=backend, weights=weights, params=params,
                states=states, current=current, reference=reference, device=device,
            )
            metrics = result_metrics(initial_cost, j16_cost[indices], final_cost)
            error = numeric_max_error(metrics, summary["records"][mode][split])
            maximum_error = max(maximum_error, error)
            replayed_metrics[mode][split] = metrics
            reports[mode][split] = {
                "max_metric_abs_error": error,
                "replayed_final_cost": metrics["final_cost"],
            }

    comparisons = {}
    conditions = []
    for split in ("train", "episode_heldout"):
        base_metrics = replayed_metrics["box1"][split]
        wide_metrics = replayed_metrics["box3"][split]
        comparison = {
            "mean_cost_improvement": (
                base_metrics["final_cost"]["mean"] - wide_metrics["final_cost"]["mean"]
            ),
            "recovery_improvement": (
                wide_metrics["headroom_recovery"] - base_metrics["headroom_recovery"]
            ),
            "p05_gain_delta": wide_metrics["gain"]["p05"] - base_metrics["gain"]["p05"],
            "regressed_fraction_delta": (
                wide_metrics["regressed_fraction"] - base_metrics["regressed_fraction"]
            ),
        }
        comparisons[split] = comparison
        conditions.extend([
            comparison["mean_cost_improvement"] >= 2.0,
            comparison["recovery_improvement"] >= 0.05,
            comparison["p05_gain_delta"] >= -0.5,
            comparison["regressed_fraction_delta"] <= 0.02,
        ])
    comparison_error = numeric_max_error(comparisons, summary["comparison"])
    paired_initial_error = float(np.max(np.abs(initial_all["box1"] - initial_all["box3"])))
    initial_contract_delta = abs(
        paired_initial_error - float(summary["contract"]["paired_initial_action_max_abs_error"])
    )
    contract_sealed = (
        summary["contract"]["formal_validation_loaded"] is False
        and summary["contract"]["test_loaded"] is False
    )
    recomputed_pass = all(conditions)
    decision_matches = (
        recomputed_pass is bool(summary["decision"]["oac_box3_integration_authorized"])
    )
    passed = (
        maximum_error <= 2e-4 and comparison_error <= 2e-4
        and initial_contract_delta <= 1e-9 and contract_sealed and decision_matches
    )
    report = {
        "qualification": (
            "DBM_TASK_LOSS_SUPPORT_FULL_AB_INDEPENDENT_VALIDATION_PASS"
            if passed else "DBM_TASK_LOSS_SUPPORT_FULL_AB_INDEPENDENT_VALIDATION_FAIL"
        ),
        "artifact_qualification": summary["qualification"],
        "maximum_metric_abs_error": maximum_error,
        "comparison_max_abs_error": comparison_error,
        "paired_initial_action_max_abs_error": paired_initial_error,
        "paired_initial_action_contract_delta": initial_contract_delta,
        "formal_validation_and_test_sealed": contract_sealed,
        "decision_matches_recomputed_gate": decision_matches,
        "records": reports,
    }
    (args.artifact / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
