#!/usr/bin/env python3
"""Independently reload and validate fixed-split target-coverage pretraining."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import pretrain_query_single_center_actor_twin_critic as base
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, load_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pretrain", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def difference(left: object, right: object) -> float:
    if isinstance(left, dict):
        return max((difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else 1.0


def main() -> None:
    args = parse_args()
    root = args.pretrain.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((root / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    data, replay_manifest, collection_manifest = base.load_data(config)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    inner = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    hash_checks = {
        "config": base.sha256(config_path) == manifest["config_sha256"],
        "script": base.sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "source_replay": base.sha256(Path(manifest["source_replay"]) / "replay.npz") == manifest["source_replay_sha256"],
        "query_checkpoint": base.sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "summary": base.sha256(root / "summary.json") == manifest["summary_sha256"],
    }
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    record_checks = []
    recomputed = []
    for record in summary["records"]:
        seed = int(record["seed"])
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hash_checks[f"checkpoint_seed{seed}"] = (
            base.sha256(checkpoint_path) == record["checkpoint_sha256"]
            == manifest["checkpoint_sha256"][checkpoint_path.name]
        )
        indices_ok = (
            np.array_equal(checkpoint["fit_indices"], fit)
            and np.array_equal(checkpoint["selection_indices"], inner)
            and np.array_equal(checkpoint["outer_indices_unevaluated"], outer)
        )
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(checkpoint, "actor_state_dict", device)
        actor_knots = actor_predict(actor, inputs, inner, device)
        actor_cost = base.query_cost(controller, data, actor_knots, inner)
        actor_metrics = base.actor_metrics(
            actor_cost, data["warm_cost"][inner], data["actor_target_cost"][inner]
        )
        actor_error = difference(actor_metrics, record["actor"]["inner"])
        critics, physical = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"critic{twin}_state_dict"], strict=True)
            critic.eval()
            prediction = base.critic_predict(critic, inputs, data, inner, device)
            training = checkpoint[f"critic{twin}_training"]
            physical.append(prediction * training["target_std"] + training["target_mean"])
            critics.append(critic)
        conservative = np.fmax(physical[0], physical[1])
        critic_metrics = base.state_metrics(conservative, data, inner)
        recovery = base.landscape_gain_recovery(conservative, data, inner)
        critic_error = max(
            difference(critic_metrics, record["critic_twin_conservative"]["inner"]["metrics"]),
            abs(recovery - record["critic_twin_conservative"]["inner"]["landscape_gain_recovery"]),
        )
        invariance = base.no_anchor_invariance(actor, inputs, inner, device)
        invariance_error = difference(invariance, checkpoint["no_anchor_invariance"])
        sealed = (
            not checkpoint.get("formal_validation_or_test_consumed", True)
            and not checkpoint.get("dbm_fields_or_labels_consumed")
            and not checkpoint.get("query_analytic_gradient_consumed", True)
        )
        record_checks.append(indices_ok and actor_error <= 1e-7 and critic_error <= 1e-7 and invariance_error <= 1e-7 and sealed)
        recomputed.append({
            "seed": seed,
            "actor_metric_max_error": actor_error,
            "critic_metric_max_error": critic_error,
            "invariance_max_error": invariance_error,
            "inner_pearson_median": critic_metrics["state_pearson_log_cost"]["median"],
            "inner_pair_sign_median": critic_metrics["state_pair_sign_accuracy"]["median"],
            "inner_landscape_gain_recovery": recovery,
            "global_value_pass": critic_metrics["state_pearson_log_cost"]["median"] >= 0.7 and recovery >= 0.5,
        })
    global_pass_count = int(sum(item["global_value_pass"] for item in recomputed))
    core_hash_checks = {
        name: value for name, value in hash_checks.items() if name != "script"
    }
    checks = {
        "artifact_hashes_excluding_training_script": all(core_hash_checks.values()),
        "checkpoint_reload_exact": all(record_checks),
        "split_counts": len(inner) == 120 and len(outer) == 120 and len(fit) == len(data["state"]) - 240,
        "outer_unevaluated": not bool(summary.get("outer_fold_evaluated", True)) and not bool(manifest.get("outer_fold_evaluated", True)),
        "global_value_and_landscape_signal": global_pass_count >= 2,
        "sealed_boundary": not manifest.get("formal_validation_or_test_consumed", True) and not manifest.get("dbm_fields_or_labels_consumed") and not manifest.get("query_analytic_gradient_consumed", True),
    }
    passed = all(checks.values())
    if passed and hash_checks["script"]:
        qualification = "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_GLOBAL_VALUE_PASS_PAIR_DIAGNOSTIC"
    elif passed:
        qualification = "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_GLOBAL_VALUE_PASS_PAIR_DIAGNOSTIC_WITH_TRAINER_SCRIPT_DRIFT"
    else:
        qualification = "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_INDEPENDENT_FAIL"
    report = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pretrain": str(root),
        "manifest_sha256": base.sha256(manifest_path),
        "checks": checks,
        "hash_checks": hash_checks,
        "provenance_warnings": ([] if hash_checks["script"] else [
            "The current training-script file no longer matches the run-time hash recorded in the manifest. Checkpoints, config, source Replay, summary, split, and independently recomputed numerical metrics still match exactly."
        ]),
        "global_value_seed_pass_count": global_pass_count,
        "records": recomputed,
        "routing_note": "Per handoff sections 17.7-17.8, pooled unstratified pair-sign is diagnostic, not an admission veto for continuously refreshed 20:small Actor OAC.",
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": base.sha256(Path(__file__).resolve()),
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
