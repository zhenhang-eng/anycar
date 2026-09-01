#!/usr/bin/env python3
"""Validate the fixed-Replay OAC-1 learning curve without new rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from run_mppi_oac1_fixed_replay_curve import (
    evaluate_checkpoint,
    load_flat_checkpoint,
    load_value_checkpoint,
)
from train_mppi_online_absolute_sac import load_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def metric_error(left: dict, right: dict) -> float:
    pairs = (
        (left["core"]["actor_visited"]["material_pair_accuracy_conservative"],
         right["core"]["actor_visited"]["material_pair_accuracy_conservative"]),
        (left["core"]["bad_action_correction"]["initially_wrong_corrected_fraction"],
         right["core"]["bad_action_correction"]["initially_wrong_corrected_fraction"]),
        (left["heldout_bank"]["material_pair_accuracy"],
         right["heldout_bank"]["material_pair_accuracy"]),
        (left["actor_visited_by_speed"]["2.8"]["material_pair_accuracy"],
         right["actor_visited_by_speed"]["2.8"]["material_pair_accuracy"]),
        (left["flat_train_calibrated_heldout"]["heldout"]["bank_best_recall"],
         right["flat_train_calibrated_heldout"]["heldout"]["bank_best_recall"]),
        (left["flat_train_calibrated_heldout"]["heldout"]["warm_false_stay"],
         right["flat_train_calibrated_heldout"]["heldout"]["warm_false_stay"]),
    )
    return max(abs(float(a) - float(b)) for a, b in pairs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    parent = Path(contract["parent_run"])
    parent_contract = json.loads((parent / "contract.json").read_text())
    outer_fold = int(contract.get("outer_fold", parent_contract["fold"]))
    bank_root = Path(contract["arguments"]["bank_root"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    checks = {
        "contract": contract["qualification"] == "OAC1_FIXED_REPLAY_CURVE_CONTRACT",
        "new_dbm_rollouts_zero": int(contract["new_dbm_rollouts"]) == 0,
        "actor_not_constructed": not contract["actor_module_constructed"],
        "actor_optimizer_not_constructed": not contract["actor_optimizer_constructed"],
        "actor_update_zero": int(summary["actor_update_count"]) == 0,
        "formal_validation_sealed": not summary["formal_validation_loaded"],
        "test_sealed": not summary["test_loaded"],
        "outer_fold_registered": outer_fold in (0, 1, 2),
        "parent_summary_hash": (
            sha256_file(parent / "summary.json") == contract["parent_summary_sha256"]
        ),
        "parent_contract_hash": (
            sha256_file(parent / "contract.json") == contract["parent_contract_sha256"]
        ),
    }
    records = []
    for row in summary["records"]:
        seed = int(row["seed"])
        replay_path = Path(row["source_replay"])
        replay_hash = sha256_file(replay_path)
        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        seed_checks = {
            "replay_hash_before_after_equal": (
                replay_hash == row["source_replay_sha256_before"]
                == row["source_replay_sha256_after"]
                == contract["source_replay_sha256"][str(seed)]
            ),
            "replay_rows_15360": len(replay["cost"]) == 15360,
        }
        point_records = []
        for point in row["points"]:
            updates = int(point["total_updates"])
            checkpoint = args.run_dir / f"seed_{seed}" / f"updates_{updates}"
            critic1, payload1 = load_value_checkpoint(checkpoint / "critic1.pt", device)
            critic2, payload2 = load_value_checkpoint(checkpoint / "critic2.pt", device)
            flat, flat_payload = load_flat_checkpoint(checkpoint / "flat_head.pt", device)
            for payload in (payload1, payload2, flat_payload):
                if int(payload["actor_update_count"]) != 0:
                    raise AssertionError("curve checkpoint contains Actor update")
                if int(payload["total_critic_update_label"]) != updates:
                    raise AssertionError("checkpoint update label mismatch")
            recomputed = evaluate_checkpoint(
                argparse.Namespace(
                    material_gap=float(contract["arguments"]["material_gap"]),
                    flat_gap=float(contract["arguments"]["flat_gap"]),
                ),
                data, folds, replay, critic1, payload1, critic2, payload2,
                flat, flat_payload, device, outer_fold,
            )
            error = metric_error(recomputed, point)
            point_records.append({
                "updates": updates, "metric_max_abs_error": error,
                "passed": error < 1e-7,
            })
        seed_checks["all_points_recompute"] = all(
            item["passed"] for item in point_records
        )
        checks[f"seed_{seed}"] = bool(all(seed_checks.values()))
        records.append({
            "seed": seed, "checks": seed_checks, "points": point_records
        })
    passed = bool(all(checks.values()))
    result = {
        "qualification": (
            "OAC1_FIXED_REPLAY_CURVE_VALIDATION_PASS" if passed
            else "OAC1_FIXED_REPLAY_CURVE_VALIDATION_FAIL"
        ),
        "checks": checks,
        "records": records,
        "passed": passed,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.run_dir / "validator_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise AssertionError("fixed-Replay curve validation failed")


if __name__ == "__main__":
    main()
