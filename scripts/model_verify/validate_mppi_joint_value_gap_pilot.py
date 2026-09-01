#!/usr/bin/env python3
"""Independently recompute the final joint Value/move-coefficient pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import AbsoluteActionValueCritic, make_folds
from train_mppi_joint_value_gap_pilot import (
    ContinuousGapHead,
    evaluate,
)
from train_mppi_online_absolute_sac import load_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path,
        nargs="?",
        default=Path(
            "outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_20260820_v3"
        ),
    )
    parser.add_argument(
        "--bank-root", type=Path,
        default=Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class EvaluationArgs:
    material_gap = 0.1
    flat_gap = 0.1
    target_mode = "move_coefficient"


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def load_final_critic(
    parent_path: Path, final_path: Path, device: torch.device,
) -> tuple[AbsoluteActionValueCritic, dict[str, Any]]:
    parent = torch.load(parent_path, map_location=device)
    final = torch.load(final_path, map_location=device)
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(final["model"], strict=True)
    return model, parent


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    if contract["arguments"]["target_mode"] != "move_coefficient":
        raise AssertionError("validator is registered for the coefficient arm")
    outer_fold = int(contract.get("outer_fold", 0))
    if outer_fold not in (0, 1, 2):
        raise AssertionError("outer fold is not registered")
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    parent_run = Path(contract["arguments"]["parent_run"])
    records = []
    for source in summary["records"]:
        seed = int(source["seed"])
        final_source = source["points"][-1]
        update = int(final_source["additional_updates"])
        parent = parent_run / f"seed_{seed}" / "updates_6400"
        final_dir = args.run_dir / f"seed_{seed}" / f"updates_{update}"
        critic1, payload1 = load_final_critic(
            parent / "critic1.pt", final_dir / "critic1.pt", device
        )
        critic2, payload2 = load_final_critic(
            parent / "critic2.pt", final_dir / "critic2.pt", device
        )
        gap_payload = torch.load(final_dir / "gap_head.pt", map_location=device)
        gap_head = ContinuousGapHead(output_mode="move_coefficient").to(device)
        gap_head.load_state_dict(gap_payload["model"], strict=True)
        replay_path = Path(source["source_replay"])
        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        recomputed = evaluate(
            EvaluationArgs(), data, folds, replay, critic1, payload1,
            critic2, payload2, gap_head, device, outer_fold,
        )
        comparisons = {
            "actor_visited_pair": close(
                recomputed["actor_visited_material_pair_accuracy"],
                final_source["actor_visited_material_pair_accuracy"],
            ),
            "heldout_bank_pair": close(
                recomputed["heldout_bank"]["material_pair_accuracy"],
                final_source["heldout_bank"]["material_pair_accuracy"],
            ),
            "gap_correlation": close(
                recomputed["gap_heldout"]["log_gap_pearson"],
                final_source["gap_heldout"]["log_gap_pearson"],
            ),
            "coefficient_mae": close(
                recomputed["gap_heldout"]["move_coefficient_mae"],
                final_source["gap_heldout"]["move_coefficient_mae"],
            ),
            "fixed_recall": close(
                recomputed["gap_heldout"]["fixed_gap_0_1"]["bank_best_recall"],
                final_source["gap_heldout"]["fixed_gap_0_1"]["bank_best_recall"],
            ),
            "fixed_false_stay": close(
                recomputed["gap_heldout"]["fixed_gap_0_1"]["warm_false_stay"],
                final_source["gap_heldout"]["fixed_gap_0_1"]["warm_false_stay"],
            ),
            "replay_hash": sha256_file(replay_path) == source["source_replay_sha256"],
            "actor_update_zero": int(gap_payload["actor_update_count"]) == 0,
            "formal_validation_sealed": not bool(gap_payload["formal_validation_loaded"]),
            "test_sealed": not bool(gap_payload["test_loaded"]),
            "optimizers_serialized": all(
                "optimizer" in torch.load(final_dir / name, map_location="cpu")
                for name in ("critic1.pt", "critic2.pt", "gap_head.pt")
            ),
        }
        fixed = recomputed["gap_heldout"]["fixed_gap_0_1"]
        mechanism_pass = (
            fixed["bank_best_recall"] >= 0.80
            and fixed["warm_false_stay"] <= 0.10
        )
        records.append({
            "seed": seed,
            "additional_updates": update,
            "checks": comparisons,
            "all_checks_passed": bool(all(comparisons.values())),
            "mechanism_pass": bool(mechanism_pass),
            "metrics": {
                "actor_visited_pair_accuracy": recomputed["actor_visited_material_pair_accuracy"],
                "heldout_bank_pair_accuracy": recomputed["heldout_bank"]["material_pair_accuracy"],
                "heldout_top1_recovery": recomputed["heldout_bank"]["top1_headroom_recovery"],
                "gap_correlation": recomputed["gap_heldout"]["log_gap_pearson"],
                "move_coefficient_mae": recomputed["gap_heldout"]["move_coefficient_mae"],
                "speed_2_8_pair_accuracy": recomputed["actor_visited_by_speed"]["2.8"]["material_pair_accuracy"],
                "lag2_bad_action_accuracy": recomputed["bad_action_correction"]["lag2_final_accuracy"],
                "initially_wrong_corrected_fraction": recomputed["bad_action_correction"]["initially_wrong_corrected_fraction"],
                **fixed,
            },
        })
    pass_count = sum(row["mechanism_pass"] for row in records)
    all_curve_points_pass = all(
        point["gap_heldout"]["fixed_gap_0_1"]["bank_best_recall"] >= 0.80
        and point["gap_heldout"]["fixed_gap_0_1"]["warm_false_stay"] <= 0.10
        for source in summary["records"]
        for point in source["points"]
        if int(point["additional_updates"]) > 0
    )
    validation_pass = (
        all(row["all_checks_passed"] for row in records)
        and pass_count >= 2
        and int(summary["actor_update_count"]) == 0
        and int(summary["new_dbm_rollouts"]) == 0
        and not bool(summary["formal_validation_loaded"])
        and not bool(summary["test_loaded"])
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "JOINT_MOVE_COEFFICIENT_VALIDATION_PASS"
            if validation_pass else "JOINT_MOVE_COEFFICIENT_VALIDATION_FAIL"
        ),
        "passed": bool(validation_pass),
        "mechanism_pass_count": int(pass_count),
        "all_12_post_training_curve_points_pass_fixed_gate": bool(all_curve_points_pass),
        "records": records,
        "note": (
            "The training summary used the generic calibrated-gap gate.  For the "
            "move-coefficient arm the pre-registered physical operating point is "
            "coefficient<=0.5, equivalent to raw gap<=0.1."
        ),
    }
    (args.run_dir / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    analysis = {
        "qualification": "JOINT_MOVE_COEFFICIENT_MECHANISM_PASS" if validation_pass else "JOINT_MOVE_COEFFICIENT_MECHANISM_FAIL",
        "fixed_physical_gate": "move coefficient <=0.5 <=> implied positive gap <=0.1",
        "actor_visited_pair_accuracy": [row["metrics"]["actor_visited_pair_accuracy"] for row in records],
        "heldout_bank_pair_accuracy": [row["metrics"]["heldout_bank_pair_accuracy"] for row in records],
        "heldout_top1_recovery": [row["metrics"]["heldout_top1_recovery"] for row in records],
        "gap_correlation": [row["metrics"]["gap_correlation"] for row in records],
        "move_coefficient_mae": [row["metrics"]["move_coefficient_mae"] for row in records],
        "bank_best_recall": [row["metrics"]["bank_best_recall"] for row in records],
        "warm_false_stay": [row["metrics"]["warm_false_stay"] for row in records],
        "speed_2_8_pair_accuracy": [row["metrics"]["speed_2_8_pair_accuracy"] for row in records],
        "lag2_bad_action_accuracy": [row["metrics"]["lag2_bad_action_accuracy"] for row in records],
        "initially_wrong_corrected_fraction": [row["metrics"]["initially_wrong_corrected_fraction"] for row in records],
        "all_12_post_training_curve_points_pass_fixed_gate": bool(all_curve_points_pass),
        "actor_update_count": 0,
        "new_dbm_rollouts": 0,
        "formal_validation_loaded": False,
        "test_loaded": False,
        "limitations": [
            (
                "outer fold 0 is a consumed mechanism audit, not fresh model selection"
                if outer_fold == 0
                else f"outer fold {outer_fold} is an independent split replication"
            ),
            "coefficient is not yet connected to Actor updates or closed-loop deployment",
            "local sensitivity/curvature is not predicted in this arm",
        ],
    }
    (args.run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not validation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
