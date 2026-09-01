#!/usr/bin/env python3
"""Calibrate and replay the TR2-B move threshold on internal-selection only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_direct_trust_alpha_policy import (
    DEFAULT_OUTPUT,
    evaluate,
    extra_tensors,
    make_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def grouped(policy, data, tensors, extra, index, train_args, device):
    output = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        one = index[np.isclose(data.reference_speed[index], speed)]
        metrics = evaluate(policy, data, tensors, extra, one, train_args, device)
        output[str(float(speed))] = {
            "context_count": len(one),
            "old_cost_mean": metrics["old_cost"]["mean"],
            "policy_cost_mean": metrics["direct_cost"]["mean"],
            "gain_mean": metrics["gain_vs_old"]["mean"],
            "gain_p05": metrics["gain_vs_old"]["p05"],
            "gain_worst": metrics["gain_vs_old"]["minimum"],
            "move_fraction": metrics["move_fraction"],
        }
    return output


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "training_summary.json").read_text())
    selected_seed = int(summary["selected_seed"])
    selected_path = args.run / f"selection_alpha_seed{selected_seed}.pt"
    selected = torch.load(selected_path, map_location="cpu")
    labels = Path(selected["labels"])
    old_path = Path(selected["old_actor"])
    for name, expected in selected["labels_hashes"].items():
        if sha256_file(labels / name) != expected:
            raise AssertionError(f"TR1 label hash mismatch: {name}")
    if sha256_file(old_path) != selected["old_actor_sha256"]:
        raise AssertionError("old Actor hash mismatch")
    old_payload = load_actor_payload(old_path)
    data, _, splits = load_dataset(labels, old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    selection_index = np.flatnonzero(np.isin(data.episodes, splits["internal_selection"]))
    train_args = argparse.Namespace(**selected["training_arguments"])
    train_args.device = args.device
    train_args.evaluation_batch_size = 128
    policy = make_policy(old_payload, device, dropout=0.0)
    policy.load_state_dict(selected["policy_state_dict"], strict=True)
    thresholds = np.concatenate((np.arange(0.50, 1.00, 0.01), np.asarray((0.995, 0.999))))
    rows = []
    for threshold in thresholds:
        train_args.move_threshold = float(threshold)
        metrics = evaluate(policy, data, tensors, extra, selection_index, train_args, device)
        gain = metrics["gain_vs_old"]
        gate = {
            "mean_positive": gain["mean"] > 0.0,
            "median_nonnegative": gain["median"] >= -1e-6,
            "p05_nonnegative": gain["p05"] >= -1e-6,
            "worst_at_least_minus_5": gain["minimum"] >= -5.0,
            "stay_recall_at_least_half": metrics["stay_recall"] >= 0.5,
        }
        rows.append({
            "threshold": float(threshold),
            "metrics": metrics,
            "gates": gate,
            "pass": all(gate.values()),
        })
    passing = [row for row in rows if row["pass"]]
    if not passing:
        raise AssertionError("no internal-selection threshold passes TR2-B gates")
    winner = min(passing, key=lambda row: row["metrics"]["direct_cost"]["mean"])
    threshold = float(winner["threshold"])
    train_args.move_threshold = threshold
    speed = grouped(policy, data, tensors, extra, selection_index, train_args, device)
    speed_gate = all(row["gain_mean"] >= 0.0 for row in speed.values())
    qualification = "TR2B_PASS_ALPHA_AC_READY" if speed_gate else "TR2B_FAIL_SPEED_GROUP"
    calibrated = copy.deepcopy(selected)
    calibrated.update({
        "qualification": qualification,
        "move_threshold": threshold,
        "threshold_calibration": "internal_selection direct DBM tail gate",
        "threshold_gates": winner["gates"],
        "test_policy": "formal validation and test not loaded or evaluated",
    })
    output_path = args.run / "trust_alpha_policy_tail_calibrated.pt"
    torch.save(calibrated, output_path)
    report = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "source_selection_checkpoint": str(selected_path.resolve()),
        "source_selection_checkpoint_sha256": sha256_file(selected_path),
        "calibrated_checkpoint": str(output_path.resolve()),
        "calibrated_checkpoint_sha256": sha256_file(output_path),
        "selected_seed": selected_seed,
        "selected_epoch": int(summary["selected_epoch"]),
        "policy_parameter_count": policy.parameter_count,
        "internal_selection_context_count": len(selection_index),
        "threshold_grid": rows,
        "selected_threshold": threshold,
        "selected_metrics": winner["metrics"],
        "by_reference_speed_mps": speed,
        "all_speed_mean_nonregression": speed_gate,
        "qualification": qualification,
        "ac_policy": (
            "ready as train-domain initialization/replay policy; formal validation "
            "and test remain sealed"
        ),
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.run / "tr2b_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected_threshold": threshold,
        "selected_metrics": winner["metrics"],
        "by_reference_speed_mps": speed,
        "checkpoint": str(output_path),
        "qualification": qualification,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
