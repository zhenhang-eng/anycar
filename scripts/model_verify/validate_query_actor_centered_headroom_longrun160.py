#!/usr/bin/env python3
"""Independently replay the 160-round Actor-centered Query headroom audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_actor_centered_headroom import pooled_metrics  # noqa: E402
from run_query_actor_centered_headroom_longrun160 import dump_json, source_contract  # noqa: E402
from run_query_forward_response_landscape_pilot import sha256  # noqa: E402
from validate_query_actor_centered_headroom import numeric_leaf_error, validate_seed  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_actor_centered_headroom_longrun160_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    if sha256(config_path) != manifest["config_sha256"]:
        raise AssertionError("config hash mismatch")
    if sha256(Path(manifest["script"])) != manifest["script_sha256"]:
        raise AssertionError("runner hash mismatch")
    if sha256(summary_path) != manifest["summary_sha256"]:
        raise AssertionError("summary hash mismatch")
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"],
            summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"],
            summary["actor_or_critic_trained"])):
        raise AssertionError("sealed boundary or no-training contract violated")

    source_summary_path, source_validation_path, source_summary, _ = source_contract(config)
    if sha256(source_summary_path) != manifest["source_actor_summary_sha256"]:
        raise AssertionError("source summary manifest hash mismatch")
    if sha256(source_validation_path) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("source validation manifest hash mismatch")
    expected_rounds = config["expected_selected_round_by_seed"]
    source_records = source_summary["records"]
    record_values = source_records.values() if isinstance(source_records, dict) else source_records
    actual_rounds = {str(record["seed"]): int(record["selected_round"])
                     for record in record_values}
    if actual_rounds != {key: int(value) for key, value in expected_rounds.items()}:
        raise AssertionError("source selected-round contract mismatch")
    adapter_reports = summary["source_cost_execution_adapter"]
    if len(adapter_reports) != len(config["actor_seeds"]):
        raise AssertionError("source execution adapter seed count mismatch")
    adapter_recompute_errors = []
    for adapter_report in adapter_reports:
        seed = int(adapter_report["seed"])
        if sha256(Path(adapter_report["original_checkpoint"])) != adapter_report["original_checkpoint_sha256"]:
            raise AssertionError(f"original checkpoint changed for seed {seed}")
        if sha256(Path(adapter_report["original_arrays"])) != adapter_report["original_arrays_sha256"]:
            raise AssertionError(f"original source arrays changed for seed {seed}")
        if sha256(Path(adapter_report["adapter_arrays"])) != adapter_report["adapter_arrays_sha256"]:
            raise AssertionError(f"source adapter arrays changed for seed {seed}")
        if adapter_report["adapter_arrays_sha256"] != manifest["source_execution_adapter_arrays_sha256"][f"seed_{seed}"]:
            raise AssertionError(f"source adapter manifest hash mismatch for seed {seed}")
        if adapter_report["actor_action_max_abs_error"] != 0.0:
            raise AssertionError(f"source Actor action adapter mismatch for seed {seed}")
        if adapter_report["cost_difference"]["maximum"] > float(
                config["source_execution_adapter"]["maximum_allowed_absolute_cost_difference"]):
            raise AssertionError(f"source cost execution-path difference too large for seed {seed}")
        if adapter_report["cost_difference"]["maximum_scaled_relative"] > float(
                config["source_execution_adapter"]["maximum_allowed_scaled_relative_cost_difference"]):
            raise AssertionError(f"source relative cost execution-path difference too large for seed {seed}")
        selected_round = int(adapter_report["selected_round"])
        with np.load(adapter_report["original_arrays"], allow_pickle=False) as original, np.load(
                adapter_report["adapter_arrays"], allow_pickle=False) as adapter:
            if not np.array_equal(original["selection_indices"], adapter["selection_indices"]):
                raise AssertionError(f"source adapter rows differ for seed {seed}")
            original_action = np.asarray(
                original["selection_round_action"][selected_round], np.float32
            )
            adapter_action = np.asarray(
                adapter["selection_round_action"][selected_round], np.float32
            )
            if not np.array_equal(original_action, adapter_action):
                raise AssertionError(f"source adapter action differs for seed {seed}")
            original_cost = np.asarray(
                original["selection_round_cost"][selected_round], np.float64
            )
            adapter_cost = np.asarray(
                adapter["selection_round_cost"][selected_round], np.float64
            )
        difference = np.abs(original_cost - adapter_cost)
        scale_floor = float(config["source_execution_adapter"]["cost_difference_scale_floor"])
        scaled = difference / np.maximum(np.abs(original_cost), scale_floor)
        recomputed = {
            "maximum": float(np.max(difference)),
            "mean": float(np.mean(difference)),
            "median": float(np.median(difference)),
            "maximum_scaled_relative": float(np.max(scaled)),
            "p95": float(np.quantile(difference, 0.95)),
            "p95_scaled_relative": float(np.quantile(scaled, 0.95)),
        }
        adapter_recompute_errors.append(
            numeric_leaf_error(adapter_report["cost_difference"], recomputed)
        )

    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    if sha256(Path(replay_manifest["query_checkpoint"])) != manifest["query_checkpoint_sha256"]:
        raise AssertionError("Query checkpoint hash mismatch")
    rows = np.flatnonzero(data["fold_id"] == int(config["split_contract"]["inner_selection_fold"]))
    if len(rows) != int(config["split_contract"]["expected_state_count"]):
        raise AssertionError("inner split size changed")
    if np.any(data["fold_id"][rows] == int(config["split_contract"]["outer_fold"])):
        raise AssertionError("outer fold leakage")

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    reports = [
        validate_seed(record, config, data, rows, controller, device)
        for record in summary["records"]
    ]
    recomputed_pooled = pooled_metrics(summary["records"], config)
    pooled_metric_error = numeric_leaf_error(summary["pooled_round_metrics"], recomputed_pooled)
    final = recomputed_pooled[-1]
    aggregate = float(final["aggregate_residual_reduction"])
    majority = float(final["improved_fraction"]) > 0.5
    if aggregate < 0.02:
        expected_decision = "STOP_BROAD_QUERY_ACTOR_EXPANSION_LOW_HEADROOM"
    elif aggregate < 0.05 or not majority:
        expected_decision = "QUERY_ACTOR_HAS_SMALL_HEADROOM_ALLOW_SINGLE_VARIABLE_AB"
    else:
        expected_decision = "QUERY_ACTOR_HAS_MATERIAL_HEADROOM_DIAGNOSE_ABSORPTION"
    checks = {
        "artifact_hashes": True,
        "longrun_actor_source_independently_qualified": True,
        "selected_rounds_and_source_hashes_exact": True,
        "source_execution_adapter_action_exact_and_cost_bounded": True,
        "source_execution_adapter_statistics_recomputed": max(adapter_recompute_errors) <= 1e-12,
        "split_and_sealed_boundary": True,
        "expected_rollout_budget": int(config["query_rollout_budget"]["total_query_rollouts"]) == 55080,
        "actor_reload_exact": all(report["max_errors"].get("actor_reload", 1.0) == 0.0 for report in reports),
        "all_55080_query_rollouts_replayed": all(report["all_array_and_query_checks_pass"] for report in reports),
        "response_fit_and_winner_chain_reconstruct": all(report["all_array_and_query_checks_pass"] for report in reports),
        "metrics_recompute": pooled_metric_error <= 1e-12 and all(
            report["round_metric_max_abs_error"] <= 1e-12 for report in reports),
        "decision_recompute": summary["decision"] == expected_decision,
    }
    passed = all(checks.values())
    report = {
        "qualification": (
            "QUERY_ACTOR_CENTERED_HEADROOM_LONGRUN160_INDEPENDENT_PASS"
            if passed else "QUERY_ACTOR_CENTERED_HEADROOM_LONGRUN160_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "seed_reports": reports,
        "pooled_metric_max_abs_error": pooled_metric_error,
        "recomputed_decision": expected_decision,
        "recomputed_final": {
            "actor_mean_cost": float(final["actor_cost"]["mean"]),
            "best_mean_cost": float(final["best_cost"]["mean"]),
            "mean_gain": float(final["gain"]["mean"]),
            "aggregate_residual_reduction": aggregate,
            "paired_median_relative_reduction": float(final["relative_reduction"]["median"]),
            "improved_fraction": float(final["improved_fraction"]),
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "actor_or_critic_trained": False,
    }
    dump_json(output / "validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
