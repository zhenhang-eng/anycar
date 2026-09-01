#!/usr/bin/env python3
"""Independently replay the train-domain TR2 Actor and its internal step scan.

The validator reads only the TR1 train-only sidecar.  It reconstructs the frozen
internal-selection Actor and final all-train refit, replays their unique direct DBM
costs, checks recorded metrics/hashes, and scans parameter interpolation using only
internal-selection episodes.  Formal validation and test are never opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_trust_region_actor import (
    DEFAULT_LABELS,
    direct_cost,
    actor_outputs,
    evaluate_actor,
    load_actor_payload,
    load_dataset,
    make_actor,
    tensorize,
)


DEFAULT_RUN = Path("outputs/mppi_proposal/direct_actor_trust_region_20260807_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--center-tolerance", type=float, default=2e-6)
    parser.add_argument("--metric-tolerance", type=float, default=1e-4)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def numeric_error(expected: Any, actual: Any) -> float:
    if isinstance(expected, dict):
        return max((numeric_error(expected[k], actual[k]) for k in expected), default=0.0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return abs(float(expected) - float(actual))
    return 0.0


def interpolate_state(
    old: dict[str, torch.Tensor], new: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in old.items():
        if torch.is_floating_point(value):
            result[name] = value + (new[name] - value) * alpha
        else:
            result[name] = value
    return result


def grouped_cost(
    actor, data, tensors, index: np.ndarray, defaults, device: torch.device
) -> dict[str, Any]:
    _, centers = actor_outputs(actor, tensors, index, defaults.evaluation_batch_size, device)
    cost = direct_cost(centers, data, tensors, index, defaults.evaluation_batch_size, device)
    old = data.old_cost[index]
    output = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        mask = np.isclose(data.reference_speed[index], speed)
        gain = old[mask] - cost[mask]
        output[str(float(speed))] = {
            "context_count": int(np.sum(mask)),
            "old_cost_mean": float(np.mean(old[mask])),
            "actor_cost_mean": float(np.mean(cost[mask])),
            "gain_mean": float(np.mean(gain)),
            "gain_p05": float(np.quantile(gain, 0.05)),
            "gain_worst": float(np.min(gain)),
        }
    return output


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "training_summary.json").read_text())
    if summary["test_policy"] != "formal validation and test not loaded or evaluated":
        raise AssertionError("TR2 run does not retain the sealed policy")
    for name, expected in summary["labels_hashes"].items():
        if sha256_file(args.labels / name) != expected:
            raise AssertionError(f"label hash mismatch: {name}")
    final_path = Path(summary["checkpoint"])
    if sha256_file(final_path) != summary["checkpoint_sha256"]:
        raise AssertionError("final checkpoint hash mismatch")
    final_payload = torch.load(final_path, map_location="cpu")
    old_path = Path(summary["old_actor"])
    old_payload = load_actor_payload(old_path)
    data, _, splits = load_dataset(args.labels, old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    # Reuse only the frozen, checkpointed selection definition.
    selected_seed = int(summary["selected_seed"])
    selection_path = args.run / f"selection_actor_seed{selected_seed}.pt"
    selection_payload = torch.load(selection_path, map_location="cpu")
    defaults = argparse.Namespace(**selection_payload["training_arguments"])
    defaults.device = args.device
    defaults.evaluation_batch_size = 128
    if selection_payload["source_labels_config_sha256"] != summary["labels_hashes"]["config.json"]:
        raise AssertionError("selection checkpoint label hash mismatch")
    selection_actor = make_actor(old_payload, device, dropout=0.0)
    selection_actor.load_state_dict(selection_payload["actor_state_dict"], strict=True)
    final_actor = make_actor(old_payload, device, dropout=0.0)
    final_actor.load_state_dict(final_payload["actor_state_dict"], strict=True)
    old_actor = make_actor(old_payload, device, dropout=0.0)
    all_index = np.arange(len(data.episodes), dtype=np.int64)
    selection_index = np.flatnonzero(np.isin(data.episodes, splits["internal_selection"]))
    _, old_centers = actor_outputs(
        old_actor, tensors, all_index, defaults.evaluation_batch_size, device
    )
    old_center_error = float(np.max(np.abs(old_centers - data.old_center)))
    if old_center_error > args.center_tolerance:
        raise AssertionError(f"old Actor reconstruction error {old_center_error}")
    selected_metrics = evaluate_actor(
        selection_actor, data, tensors, selection_index, defaults, device
    )
    final_metrics = evaluate_actor(final_actor, data, tensors, all_index, defaults, device)
    selected_error = numeric_error(summary["selected_internal_metrics"], selected_metrics)
    final_error = numeric_error(summary["final_all_train_metrics"], final_metrics)
    if selected_error > args.metric_tolerance or final_error > args.metric_tolerance:
        raise AssertionError(f"metric replay mismatch {selected_error}/{final_error}")
    scan = []
    old_state = old_payload["actor_state_dict"]
    new_state = selection_payload["actor_state_dict"]
    for alpha in np.linspace(0.0, 1.0, 21):
        actor = make_actor(old_payload, device, dropout=0.0)
        actor.load_state_dict(interpolate_state(old_state, new_state, float(alpha)), strict=True)
        metrics = evaluate_actor(actor, data, tensors, selection_index, defaults, device)
        gain = metrics["gain_vs_old"]
        tail_pass = (
            gain["median"] >= -1e-3
            and gain["p05"] >= -1e-3
            and gain["minimum"] >= -5.0 - 1e-3
        )
        scan.append({
            "alpha": float(alpha),
            "direct_cost_mean": metrics["direct_cost"]["mean"],
            "gain_mean": gain["mean"],
            "gain_median": gain["median"],
            "gain_p05": gain["p05"],
            "gain_worst": gain["minimum"],
            "stay_recall": metrics["stay_recall"],
            "tail_pass": tail_pass,
        })
    nonzero_tail_pass = [row for row in scan if row["alpha"] > 0 and row["tail_pass"]]
    selected_gain = selected_metrics["gain_vs_old"]
    gates = {
        "mean_cost_improves": selected_gain["mean"] > 0.0,
        "target_fit_improves_from_old": (
            selected_metrics["target_sigma_rmse"]
            < summary["seeds"][0]["history"][0]["target_sigma_rmse"]
        ),
        "stay_behavior_learned": selected_metrics["stay_recall"] > 0.0,
        "trust_bound_exact": selected_metrics["trust_violation_fraction"] == 0.0,
        "p05_nonnegative": selected_gain["p05"] >= 0.0,
        "worst_at_least_minus_5": selected_gain["minimum"] >= -5.0,
        "nonzero_parameter_step_tail_pass": bool(nonzero_tail_pass),
    }
    qualification = "TR2_PASS" if all(gates.values()) else "TR2_FAIL_STAY_TRUST_AND_TAIL"
    report = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": sha256_file(final_path),
        "actor_parameter_count": final_actor.parameter_count,
        "validated_contexts": len(data.episodes),
        "validated_internal_selection_contexts": len(selection_index),
        "maximum_old_center_reconstruction_error": old_center_error,
        "maximum_selected_metric_error": selected_error,
        "maximum_final_metric_error": final_error,
        "selected_internal_metrics": selected_metrics,
        "selected_by_reference_speed_mps": grouped_cost(
            selection_actor, data, tensors, selection_index, defaults, device
        ),
        "internal_parameter_step_scan": scan,
        "gates": gates,
        "qualification": qualification,
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    if args.write_report:
        (args.run / "tr2_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
