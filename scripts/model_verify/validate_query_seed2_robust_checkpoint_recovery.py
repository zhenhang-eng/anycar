#!/usr/bin/env python3
"""Independently validate the recovered seed-2 risk-selected checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import pretrain_query_single_center_actor_twin_critic as pretrain
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from recover_query_seed2_robust_checkpoint import select_round
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, direct_cost, load_inputs, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def metrics(cost: np.ndarray, warm: np.ndarray) -> dict[str, float]:
    gain = warm.astype(np.float64) - cost.astype(np.float64)
    return {
        "cost_mean": float(np.mean(cost)),
        "cost_median": float(np.median(cost)),
        "warm_win_or_tie_fraction": float(np.mean(gain >= 0)),
        "warm_gain_median": float(np.median(gain)),
        "warm_gain_p05": float(np.quantile(gain, 0.05)),
        "warm_gain_worst": float(np.min(gain)),
    }


def main() -> None:
    args = parse_args()
    root = args.recovery.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    source_root = Path(config["source_run"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    source_validation = json.loads((source_root / "validation.json").read_text())
    source_summary = json.loads((source_root / "summary.json").read_text())
    source_record = next(
        value for value in source_summary["records"] if int(value["seed"]) == int(config["seed"])
    )
    with np.load(source_record["arrays"], allow_pickle=False) as archive:
        source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    checkpoint_path = Path(summary["robust_checkpoint"])
    arrays_path = Path(summary["arrays"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with np.load(arrays_path, allow_pickle=False) as archive:
        recovery_arrays = {name: np.asarray(archive[name]) for name in archive.files}

    algorithm = source_summary["contract"]
    loader_config = {"outputs": {"absolute_replay": algorithm["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    inner = source_arrays["selection_indices"]
    rule = config["selection_rule"]
    target_mask = (
        (data["speed_kph"][inner] == int(rule["target_speed_kph"]))
        & (data["variant_index"][inner] == int(rule["target_variant_index"]))
    )
    derived_round, derived_report = select_round(
        source_arrays["selection_round_cost"], data["warm_cost"][inner],
        target_mask, float(rule["mean_cost_tolerance_above_minimum"]),
    )
    report_error = max(
        abs(float(derived_report[key]) - float(summary["selection_report"][key]))
        for key in derived_report
    )

    device = torch.device(args.device)
    actor = actor_from_payload(checkpoint, "selected_actor_state_dict", device)
    inputs = load_inputs(data, checkpoint["actor_normalization"])
    action = actor_predict(actor, inputs, inner, device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in algorithm["cost_weights"].items()}
    cost = direct_cost(controller, data, inner, action, weights)
    source_action = source_arrays["selection_round_action"][derived_round]
    source_cost = source_arrays["selection_round_cost"][derived_round]
    action_error = float(np.max(np.abs(action - source_action)))
    cost_error = float(np.max(np.abs(cost - source_cost)))

    critic_reload = True
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        try:
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
        except Exception:
            critic_reload = False
    hashes = {
        "config": sha256(config_path) == manifest["config_sha256"],
        "script": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "shared_training_implementation": (
            sha256(Path(manifest["shared_training_implementation"]))
            == manifest["shared_training_implementation_sha256"]
        ),
        "source_manifest": sha256(source_root / "manifest.json") == manifest["source_manifest_sha256"],
        "source_summary": sha256(source_root / "summary.json") == manifest["source_summary_sha256"],
        "source_arrays": sha256(Path(source_record["arrays"])) == manifest["source_arrays_sha256"],
        "summary": sha256(root / "summary.json") == manifest["summary_sha256"],
        "robust_checkpoint": sha256(checkpoint_path) == manifest["robust_checkpoint_sha256"],
        "arrays": sha256(arrays_path) == manifest["arrays_sha256"],
    }
    checks = {
        "artifact_hashes": all(hashes.values()),
        "source_run_independently_validated": (
            source_validation["qualification"]
            == "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS"
        ),
        "selection_rule_reconstructed": (
            derived_round == int(summary["target_round"]) == int(checkpoint["selected_round"])
            and report_error <= 1e-12
        ),
        "saved_recovery_arrays_exact": (
            np.array_equal(recovery_arrays["source_inner_action"], source_action)
            and np.array_equal(recovery_arrays["source_inner_cost"], source_cost)
            and np.array_equal(recovery_arrays["recovered_inner_action"], source_action)
            and np.array_equal(recovery_arrays["recovered_inner_cost"], source_cost)
            and np.array_equal(recovery_arrays["target_mask"], target_mask)
        ),
        "actor_and_query_replay_exact": action_error <= 1e-7 and cost_error <= 1e-6,
        "twin_critic_reload": critic_reload,
        "split_exact": (
            np.array_equal(checkpoint["selection_indices"], inner)
            and len(checkpoint["outer_indices_unevaluated"]) == 120
        ),
        "outer_unevaluated": (
            not summary.get("outer_fold_evaluated", True)
            and not manifest.get("outer_fold_evaluated", True)
            and not checkpoint.get("formal_validation_or_test_consumed", True)
        ),
        "sealed_boundary": (
            not checkpoint.get("dbm_fields_or_labels_consumed")
            and not checkpoint.get("query_analytic_gradient_consumed", True)
        ),
    }
    passed = all(checks.values())
    result = {
        "qualification": (
            "QUERY_SEED2_ROBUST_CHECKPOINT_RECOVERY_INDEPENDENT_TRAIN_SIDE_PASS"
            if passed else "QUERY_SEED2_ROBUST_CHECKPOINT_RECOVERY_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recovery": str(root),
        "checks": checks,
        "hash_checks": hashes,
        "selected_round": derived_round,
        "selection_report": derived_report,
        "maximum_errors": {
            "selection_report": report_error,
            "actor_action": action_error,
            "query_cost": cost_error,
        },
        "inner": metrics(cost, data["warm_cost"][inner]),
        "inner_100:3": metrics(cost[target_mask], data["warm_cost"][inner][target_mask]),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
