#!/usr/bin/env python3
"""Independent replay validator for the train-only K=1/K=4 Actor pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_multi_candidate_actor import critic_select, rollout_candidates


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
    if summary["contract"]["formal_validation_loaded"] or summary["contract"]["test_loaded"]:
        raise AssertionError("sealed split contract violated")
    with np.load(args.bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.artifact / "oof_predictions.npz", allow_pickle=False) as loaded:
        oof = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, args.gt_v1
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    device = torch.device(args.device)
    checks = {}
    for count in (1, 4):
        prefix = f"k{count}_"
        index = oof[prefix + "state_index"]
        candidates = oof[prefix + "candidates"]
        replay = rollout_candidates(
            backend, weights, params, candidates, states[index], current[index],
            reference[index], 128, device,
        )
        stored = oof[prefix + "candidate_cost"]
        replay_error = float(np.max(np.abs(replay - stored)))
        oracle_error = float(np.max(np.abs(
            replay.min(1) - oof[prefix + "oracle_selected_cost"]
        )))
        critic_errors = []
        for seed in np.unique(oof[prefix + "seed"]):
            for fold in np.unique(oof[prefix + "fold"]):
                mask = (
                    (oof[prefix + "seed"] == seed)
                    & (oof[prefix + "fold"] == fold)
                )
                choice = critic_select(
                    args.bank_root, int(seed), int(fold), data,
                    candidates[mask], index[mask], device,
                )
                selected = replay[mask][np.arange(np.sum(mask)), choice]
                critic_errors.append(float(np.max(np.abs(
                    selected - oof[prefix + "critic_selected_cost"][mask]
                ))))
        checks[f"K{count}"] = {
            "candidate_cost_max_abs_error": replay_error,
            "oracle_selected_max_abs_error": oracle_error,
            "critic_selected_max_abs_error": max(critic_errors),
        }
    passed = all(
        max(value.values()) <= 1e-4 for value in checks.values()
    )
    report = {
        "qualification": (
            "MULTI_CANDIDATE_ACTOR_VALIDATION_PASS" if passed
            else "MULTI_CANDIDATE_ACTOR_VALIDATION_FAIL"
        ),
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.artifact / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report["qualification"], json.dumps(checks))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
