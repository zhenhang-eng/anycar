#!/usr/bin/env python3
"""Independent validator for the frozen-Actor Critic capacity/pair-delta A/B."""

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
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from run_mppi_oac_critic_capacity_pairdelta_ab import (
    ConfigurableAbsoluteActionValueCritic,
    model_for_arm,
)
from train_mppi_online_absolute_sac import load_bank, rollout_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path,
        nargs="?",
        default=Path(
            "outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def close(left, right, tolerance=1e-6):
    return bool(abs(float(left) - float(right)) <= tolerance)


def correlation(left, right):
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pair_accuracy(prediction, cost, gap=0.1):
    correct = []
    for pred, truth in zip(prediction, cost):
        i, j = np.triu_indices(len(truth), 1)
        material = np.abs(truth[i] - truth[j]) >= gap
        correct.extend((
            np.sign(pred[i][material] - pred[j][material])
            == np.sign(truth[i][material] - truth[j][material])
        ).tolist())
    return float(np.mean(correct))


def top1_regret(prediction, cost):
    chosen = cost[np.arange(len(cost)), np.argmin(prediction, axis=1)]
    return float(np.mean(chosen - cost.min(axis=1)))


def main() -> None:
    args = parse_args()
    run = args.run_dir
    contract = json.loads((run / "contract.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    with np.load(run / "predictions.npz", allow_pickle=False) as loaded:
        predictions = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(run / "outer_heldout_actor_candidates.npz", allow_pickle=False) as loaded:
        heldout_actor = {key: np.asarray(loaded[key]) for key in loaded.files}

    bank_root = Path(contract["arguments"]["bank_root"])
    gt_v1 = Path(contract["arguments"]["gt_v1"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    outer_fold = int(contract["outer_fold"])
    heldout = np.flatnonzero(folds == outer_fold)
    train = np.flatnonzero(folds != outer_fold)
    checks = {
        "qualification_contract": contract["qualification"]
        == "OAC_CRITIC_CAPACITY_PAIRDELTA_AB_CONTRACT",
        "formal_validation_sealed": not bool(summary["formal_validation_loaded"]),
        "test_sealed": not bool(summary["test_loaded"]),
        "actor_update_zero": int(summary["actor_update_count"]) == 0,
        "heldout_exact_fold": np.array_equal(heldout_actor["state_index"], heldout),
        "prediction_state_exact_fold": np.array_equal(
            predictions["heldout_state_index"], heldout
        ),
        "candidate_bank_hash": sha256_file(bank_root / "candidate_bank.npz")
        == contract["source_hashes"]["candidate_bank"],
    }
    schedules: dict[int, set[str]] = {}
    parameter_counts = {}
    model_contract_checks = []
    metric_errors = []
    for record in summary["records"]:
        seed, arm = int(record["seed"]), str(record["arm"])
        schedules.setdefault(seed, set()).add(record["training_schedule_sha256"])
        parameter_counts[arm] = int(record["parameter_count"])
        path = Path(record["checkpoint"])
        payload = torch.load(path, map_location="cpu")
        model = model_for_arm(arm)
        incompatible = model.load_state_dict(payload["model"], strict=True)
        model_contract_checks.append(
            not incompatible.missing_keys and not incompatible.unexpected_keys
            and int(payload["actor_update_count"]) == 0
            and int(sum(p.numel() for p in model.parameters()))
            == int(record["parameter_count"])
        )
        key = f"seed{seed}_{arm}"
        absolute = predictions[f"{key}_absolute_bank"]
        expected = record["evaluation"]["absolute_bank"]
        metric_errors.extend((
            abs(pair_accuracy(absolute, data["costs"][heldout])
                - float(expected["material_pair_accuracy"])),
            abs(top1_regret(absolute, data["costs"][heldout])
                - float(expected["top1_regret_mean"])),
            abs(correlation(absolute, np.log1p(data["costs"][heldout]))
                - float(expected["pearson_log_value"])),
        ))
        if arm == "pair_delta":
            device = torch.device(args.device)
            model = model.to(device).eval()
            batch = min(16, len(heldout))
            left = torch.from_numpy(data["actions"][heldout[:batch], 0]).to(device)
            history = torch.zeros((batch, 250, 7), device=device)
            reference = torch.zeros((batch, 50, 5), device=device)
            current = torch.zeros((batch, 4), device=device)
            with torch.no_grad():
                zero = model.pair_delta(history, reference, current, left, left)
                anti_left = model.pair_delta(
                    history, reference, current, left,
                    torch.from_numpy(data["actions"][heldout[:batch], 1]).to(device),
                )
                anti_right = model.pair_delta(
                    history, reference, current,
                    torch.from_numpy(data["actions"][heldout[:batch], 1]).to(device),
                    left,
                )
            model_contract_checks.append(
                float(zero.abs().max()) <= 1e-7
                and float((anti_left + anti_right).abs().max()) <= 1e-6
            )
    checks.update({
        "same_schedule_all_arms": all(len(values) == 1 for values in schedules.values()),
        "base_exact_current_parameter_count": parameter_counts.get("base") == 467585,
        "wide_parameter_ratio_1_8_to_2_2": 1.8 <= (
            parameter_counts.get("wide", 0) / max(parameter_counts.get("base", 1), 1)
        ) <= 2.2,
        "model_checkpoint_contract": bool(all(model_contract_checks)),
        "saved_metrics_recompute": max(metric_errors, default=0.0) <= 1e-5,
    })

    # Independently replay all 3 roles for all seeds on the outer-heldout set.
    device = torch.device(args.device)
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    cost_errors = []
    action = heldout_actor["actions"]
    cost = heldout_actor["costs"]
    for seed_position in range(action.shape[0]):
        for role_position in range(action.shape[1]):
            replayed = rollout_bank(
                backend, weights, params,
                action[seed_position, role_position, :, None],
                states, current, references, heldout,
                args.batch_size, device,
            )[:, 0]
            cost_errors.append(float(np.max(np.abs(
                replayed - cost[seed_position, role_position]
            ))))
    checks["heldout_dbm_cost_replay_le_1e_5"] = max(cost_errors) <= 1e-5
    checks["fixed_replay_train_only"] = True
    for seed in schedules:
        replay_path = Path(contract["arguments"]["run_dir"]) / f"seed_{seed}" / "actor_visited_replay.npz"
        with np.load(replay_path, allow_pickle=False) as loaded:
            if not np.all(np.isin(loaded["state_index"], train)):
                checks["fixed_replay_train_only"] = False

    passed = bool(all(checks.values()))
    report = {
        "passed": passed,
        "checks": checks,
        "max_saved_metric_error": max(metric_errors, default=0.0),
        "max_heldout_dbm_cost_replay_error": max(cost_errors),
        "parameter_counts": parameter_counts,
        "wide_to_base_parameter_ratio": (
            parameter_counts.get("wide", 0) / max(parameter_counts.get("base", 1), 1)
        ),
    }
    (run / "validator_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        raise AssertionError(report)
    summary["qualification"] = "OAC_CRITIC_CAPACITY_PAIRDELTA_AB_VALIDATED"
    summary["validator"] = str((run / "validator_report.json").resolve())
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
