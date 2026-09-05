#!/usr/bin/env python3
"""Independently validate the dynamic-history seed-2 OAC train-side screen."""

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
from query_batched_direct_cost import batched_direct_cost
from run_query_oac_gamma1_k_scan import warm_relative_metrics
from run_query_single_center_oac20to1 import actor_from_payload, load_inputs, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def metric_error(left: object, right: object) -> float:
    if isinstance(left, dict):
        return max((metric_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else 1.0


def main() -> None:
    args = parse_args()
    root = args.run.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    inner = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    helper = Path(__file__).resolve().parent / "query_batched_direct_cost.py"
    hashes = {
        "config": sha256(config_path) == manifest["config_sha256"],
        "script": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "shared_training_implementation": (
            sha256(Path(manifest["shared_training_implementation"]))
            == manifest["shared_training_implementation_sha256"]
        ),
        "batched_direct_cost_helper": sha256(helper) == manifest["batched_direct_cost_helper_sha256"],
        "source_replay": (
            sha256(Path(manifest["source_replay"]) / "replay.npz")
            == manifest["source_replay_sha256"]
        ),
        "summary": sha256(root / "summary.json") == manifest["summary_sha256"],
    }
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    records = []
    all_exact = True
    for source_record in summary["records"]:
        seed = int(source_record["seed"])
        arrays_path = Path(source_record["arrays"])
        checkpoint_path = Path(source_record["checkpoint"])
        hashes[f"arrays_seed{seed}"] = sha256(arrays_path) == source_record["arrays_sha256"]
        hashes[f"checkpoint_seed{seed}"] = sha256(checkpoint_path) == source_record["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hashes[f"source_actor_seed{seed}"] = (
            sha256(Path(checkpoint["source_actor_checkpoint"]))
            == checkpoint["source_actor_checkpoint_sha256"]
        )
        hashes[f"source_critic_seed{seed}"] = (
            sha256(Path(checkpoint["source_critic_checkpoint"]))
            == checkpoint["source_critic_checkpoint_sha256"]
        )
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        split_exact = (
            np.array_equal(arrays["fit_indices"], fit)
            and np.array_equal(arrays["selection_indices"], inner)
            and np.array_equal(arrays["outer_indices_unevaluated"], outer)
            and np.array_equal(checkpoint["fit_indices"], fit)
            and np.array_equal(checkpoint["selection_indices"], inner)
            and np.array_equal(checkpoint["outer_indices_unevaluated"], outer)
        )
        rounds = int(config["pilot"]["rounds"])
        online_shape_exact = (
            len(arrays["online_cost"])
            == rounds * int(config["pilot"]["fit_contexts_visited_per_round"])
            * int(config["pilot"]["candidates_per_visit"])
            and np.all(np.isin(arrays["online_state_index"], fit))
            and arrays["selection_round_cost"].shape == (rounds + 1, len(inner))
            and arrays["selection_round_action"].shape == (rounds + 1, len(inner), 8, 2)
            and np.isfinite(arrays["online_cost"]).all()
        )
        selected_round = int(np.argmin(arrays["selection_round_cost"].mean(axis=1)))
        selected_round_exact = selected_round == int(source_record["selected_round"])
        actor_payload = {
            "actor_training": checkpoint["actor_training"],
            "selected_actor_state_dict": checkpoint["selected_actor_state_dict"],
        }
        actor = actor_from_payload(actor_payload, "selected_actor_state_dict", device)
        actor_inputs = load_inputs(data, checkpoint["actor_normalization"])
        actor.eval()
        with torch.no_grad():
            tensors = tuple(torch.from_numpy(value[inner]).to(device) for value in actor_inputs)
            inner_action = actor(*tensors)[1].cpu().numpy().astype(np.float32)
        inner_cost = batched_direct_cost(controller, data, inner, inner_action, weights)
        stored_action = arrays["selection_round_action"][selected_round]
        stored_cost = arrays["selection_round_cost"][selected_round]
        action_error = float(np.max(np.abs(inner_action - stored_action)))
        cost_error = float(np.max(np.abs(inner_cost - stored_cost)))
        metrics = warm_relative_metrics(
            inner_cost, data["warm_cost"][inner], data["speed_kph"][inner],
            data["variant_index"][inner],
        )
        metrics_error = metric_error(metrics, source_record["selected"]["inner"])
        sealed = (
            not checkpoint.get("formal_validation_or_test_consumed", True)
            and not checkpoint.get("dbm_fields_or_labels_consumed")
            and not checkpoint.get("query_analytic_gradient_consumed", True)
        )
        exact = (
            split_exact and online_shape_exact and selected_round_exact and sealed
            and action_error <= 1e-7 and cost_error <= 1e-6 and metrics_error <= 1e-7
        )
        all_exact &= exact
        records.append({
            "seed": seed,
            "split_exact": bool(split_exact),
            "online_shape_exact": bool(online_shape_exact),
            "selected_round": selected_round,
            "selected_round_exact": bool(selected_round_exact),
            "selected_actor_action_max_error": action_error,
            "selected_actor_query_cost_max_error": cost_error,
            "selected_inner_metric_max_error": metrics_error,
            "sealed_boundary": bool(sealed),
            "selected_inner_mean_cost": float(inner_cost.mean()),
        })
    checks = {
        "artifact_hashes": all(hashes.values()),
        "split_counts": len(fit) == len(data["state"]) - 240 and len(inner) == 120 and len(outer) == 120,
        "single_registered_seed": [record["seed"] for record in records] == [2],
        "selected_actor_and_query_recompute_exact": bool(all_exact),
        "outer_unevaluated": (
            not summary.get("outer_fold_evaluated", True)
            and not manifest.get("outer_fold_evaluated", True)
            and all(
                "oof_cost" not in np.load(record["arrays"], allow_pickle=False).files
                and "outer_cost" not in np.load(record["arrays"], allow_pickle=False).files
                for record in summary["records"]
            )
        ),
        "sealed_boundary": (
            not manifest.get("formal_validation_or_test_consumed", True)
            and not manifest.get("dbm_fields_or_labels_consumed")
            and not manifest.get("query_analytic_gradient_consumed", True)
        ),
        "train_side_screen_progress": summary["decision"] == "DYNAMIC_HISTORY_SEED2_SCREEN_PROGRESS",
    }
    passed = all(checks.values())
    report = {
        "qualification": (
            "QUERY_DYNAMIC_HISTORY_SEED2_OAC_INDEPENDENT_TRAIN_SIDE_SCREEN_PASS"
            if passed else "QUERY_DYNAMIC_HISTORY_SEED2_OAC_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(root),
        "checks": checks,
        "hash_checks": hashes,
        "records": records,
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
