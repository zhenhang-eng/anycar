#!/usr/bin/env python3
"""Independently replay and validate the Query gamma-1 matched-output-step audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from run_query_actor_aggregation_microstep import (  # noqa: E402
    actor_gradient,
    calibrate_step,
    objective_weights,
    predict_conservative_log,
)
from run_query_oac_gamma1_matched_step import route, warm_relative_report  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    direct_cost,
    distribution,
    load_inputs,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_matched_step_20260902_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64))))


def numeric_structure_error(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max((numeric_structure_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list):
        if len(left) != len(right):
            return float("inf")
        return max((numeric_structure_error(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((output / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    source = Path(manifest["source_gamma1_k_scan"])
    replay_dir = Path(manifest["absolute_replay"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    sigma = np.asarray(config["step_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    weights_cost = {name: float(value) for name, value in config["cost_weights"].items()}
    gamma = float(config["actor_objective"]["gamma"])
    weight_maximum = float(config["actor_objective"]["weight_maximum"])
    batch_size = int(config["gradient_contract"]["batch_size"])

    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "validator_hash": sha256(Path(__file__).resolve()) == manifest["validator_sha256"],
        "helper_hashes": all(sha256(Path(path)) == value for path, value in manifest["helper_sha256"].items()),
        "summary_hash": sha256(output / "summary.json") == manifest["summary_sha256"],
        "source_manifest_hash": sha256(source / "manifest.json") == manifest["source_manifest_sha256"],
        "source_summary_hash": sha256(source / "summary.json") == manifest["source_summary_sha256"],
        "source_validation_hash": sha256(source / "validation.json") == manifest["source_validation_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "query_checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "no_new_training": not bool(summary["new_training"]) and not bool(manifest["new_training"]),
        "oof_not_evaluated": not bool(summary["oof_evaluated"]) and not bool(manifest["oof_evaluated"]),
        "formal_test_sealed": not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(manifest["query_analytic_gradient_consumed"]),
    }
    errors = {
        "base_fit_action": 0.0,
        "base_selection_action": 0.0,
        "base_selection_cost": 0.0,
        "conservative_log": 0.0,
        "weight": 0.0,
        "gradient": 0.0,
        "step_multiplier": 0.0,
        "step_achieved": 0.0,
        "step_action": 0.0,
        "step_cost": 0.0,
        "step_metrics": 0.0,
        "forbidden_actor_input": 0.0,
    }
    route_records = []
    reports = []
    recomputed_pooled_base: list[list[np.ndarray]] = [[] for _ in steps]
    recomputed_pooled_warm: list[list[np.ndarray]] = [[] for _ in steps]

    for record in summary["records"]:
        seed = int(record["seed"])
        arrays_path = output / f"seed_{seed}.npz"
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        checkpoint = torch.load(record["source_checkpoint"], map_location=device, weights_only=False)
        fit = saved["fit_indices"]
        selection = saved["selection_indices"]
        oof = saved["oof_indices_untouched"]
        seed_checks = {
            "arrays_hash": sha256(arrays_path) == record["output_arrays_sha256"] == manifest["seed_arrays_sha256"][str(seed)],
            "source_checkpoint_hash": sha256(Path(record["source_checkpoint"])) == record["source_checkpoint_sha256"],
            "source_arrays_hash": sha256(Path(record["source_arrays"])) == record["source_arrays_sha256"],
            "split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "split_indices_match_checkpoint": bool(
                np.array_equal(fit, checkpoint["fit_indices"])
                and np.array_equal(selection, checkpoint["selection_indices"])
                and np.array_equal(oof, checkpoint["oof_indices"])
            ),
            "split_episode_disjoint": not bool(
                set(data["episode_id"][fit].tolist()) & set(data["episode_id"][selection].tolist())
                or set(data["episode_id"][fit].tolist()) & set(data["episode_id"][oof].tolist())
                or set(data["episode_id"][selection].tolist()) & set(data["episode_id"][oof].tolist())
            ),
            "selected_round_match": int(checkpoint["selected_round"]) == int(record["source_selected_round"]),
            "query_rollout_count": int(record["new_query_rollouts"]) == len(selection) * (1 + len(steps)),
            "oof_not_evaluated": not bool(record["oof_evaluated"]),
        }
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(
            {
                "actor_training": checkpoint["actor_training"],
                "selected_actor_state_dict": checkpoint["selected_actor_state_dict"],
            },
            "selected_actor_state_dict",
            device,
        )
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        base_state = copy.deepcopy(actor.state_dict())
        base_fit_action = actor_predict(actor, inputs, fit, device)
        base_selection_action = actor_predict(actor, inputs, selection, device)
        base_selection_cost = direct_cost(controller, data, selection, base_selection_action, weights_cost)
        errors["base_fit_action"] = max(errors["base_fit_action"], max_error(base_fit_action, saved["base_fit_action"]))
        errors["base_selection_action"] = max(errors["base_selection_action"], max_error(base_selection_action, saved["base_selection_action"]))
        errors["base_selection_cost"] = max(errors["base_selection_cost"], max_error(base_selection_cost, saved["base_selection_cost"]))
        base_report = warm_relative_report(
            base_selection_cost, base_selection_cost,
            data["warm_cost"][selection], data["speed_kph"][selection],
        )
        errors["step_metrics"] = max(errors["step_metrics"], numeric_structure_error(base_report, record["base_inner"]))
        conservative_log = predict_conservative_log(actor, critics, trainings, inputs, fit, device, batch_size)
        weight = objective_weights(conservative_log, gamma, weight_maximum)
        gradient, names, offsets = actor_gradient(
            actor, critics, trainings, inputs, fit, weight, device, batch_size
        )
        errors["conservative_log"] = max(errors["conservative_log"], max_error(conservative_log, saved["fit_conservative_log_cost"]))
        errors["weight"] = max(errors["weight"], max_error(weight, saved["fit_weight"]))
        errors["gradient"] = max(errors["gradient"], max_error(gradient, saved["actor_parameter_gradient"]))
        seed_checks["parameter_layout"] = names == saved["parameter_names"].tolist() and np.array_equal(offsets, saved["parameter_offsets"])
        local_route_record = copy.deepcopy(record)
        local_route_record["step_gain_arrays"] = []
        local_route_record["step_warm_gain_arrays"] = []
        for step_index, target in enumerate(steps):
            actor.load_state_dict(base_state, strict=True)
            multiplier, achieved, _ = calibrate_step(
                actor, base_state, base_fit_action, inputs, fit, sigma,
                gradient, names, offsets, float(target), device,
            )
            action = actor_predict(actor, inputs, selection, device)
            cost = direct_cost(controller, data, selection, action, weights_cost)
            step_report = warm_relative_report(
                cost, base_selection_cost,
                data["warm_cost"][selection], data["speed_kph"][selection],
            )
            errors["step_multiplier"] = max(errors["step_multiplier"], abs(multiplier - float(saved["matched_step_parameter_multiplier"][step_index])))
            errors["step_achieved"] = max(errors["step_achieved"], abs(achieved - float(saved["matched_step_achieved"][step_index])))
            errors["step_action"] = max(errors["step_action"], max_error(action, saved["matched_step_selection_action"][step_index]))
            errors["step_cost"] = max(errors["step_cost"], max_error(cost, saved["matched_step_selection_cost"][step_index]))
            errors["step_metrics"] = max(errors["step_metrics"], numeric_structure_error(step_report, record["steps"][step_index]["inner"]))
            gain_base = base_selection_cost - cost
            gain_warm = data["warm_cost"][selection] - cost
            local_route_record["step_gain_arrays"].append(gain_base.tolist())
            local_route_record["step_warm_gain_arrays"].append(gain_warm.tolist())
            recomputed_pooled_base[step_index].append(gain_base)
            recomputed_pooled_warm[step_index].append(gain_warm)
        actor.load_state_dict(base_state, strict=True)
        rows = selection[:20]
        tensors = [torch.from_numpy(value[rows]).to(device) for value in inputs]
        with torch.no_grad():
            reference = actor(*tensors)[1]
            for index in (3, 4, 5):
                changed = list(tensors)
                changed[index] = torch.randn_like(changed[index])
                errors["forbidden_actor_input"] = max(
                    errors["forbidden_actor_input"],
                    float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu()),
                )
        route_records.append(local_route_record)
        reports.append({"seed": seed, "checks": seed_checks, "passed": bool(all(seed_checks.values()))})

    decision, routing_checks = route(route_records, steps)
    checks["decision_reconstructed"] = decision == summary["decision"] == manifest["decision"]
    checks["routing_reconstructed"] = numeric_structure_error(routing_checks, summary["routing_checks"]) <= 1e-7
    for step_index, target in enumerate(steps):
        key = str(float(target))
        gain_base = np.concatenate(recomputed_pooled_base[step_index]).astype(np.float64)
        gain_warm = np.concatenate(recomputed_pooled_warm[step_index]).astype(np.float64)
        expected = {
            "gain_vs_k4_base": distribution(gain_base),
            "gain_vs_warm": distribution(gain_warm),
            "positive_mean_seed_count": int(sum(
                record["steps"][step_index]["inner"]["gain_vs_round0"]["mean"] > 0.0
                for record in summary["records"]
            )),
        }
        checks[f"pooled_step_{key}"] = numeric_structure_error(expected, summary["pooled_steps"][key]) <= 1e-7

    tolerances = {
        "base_fit_action": 1e-6,
        "base_selection_action": 1e-6,
        "base_selection_cost": 1e-6,
        "conservative_log": 1e-6,
        "weight": 1e-6,
        "gradient": 2e-6,
        "step_multiplier": 1e-9,
        "step_achieved": 1e-7,
        "step_action": 1e-6,
        "step_cost": 1e-6,
        "step_metrics": 1e-6,
        "forbidden_actor_input": 1e-7,
    }
    for name, tolerance in tolerances.items():
        checks[f"{name}_within_tolerance"] = errors[name] <= tolerance
    checks["all_seed_checks"] = all(report["passed"] for report in reports)
    passed = bool(all(checks.values()))
    qualification = (
        "QUERY_OAC_GAMMA1_MATCHED_STEP_INDEPENDENT_PASS"
        if passed else "QUERY_OAC_GAMMA1_MATCHED_STEP_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "maximum_absolute_errors": errors,
        "tolerances": tolerances,
        "seed_reports": reports,
        "recomputed_routing_checks": routing_checks,
        "new_training": False,
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if passed:
        manifest["qualification"] = qualification
        manifest["validation"] = str(validation_path)
        manifest["validation_sha256"] = sha256(validation_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "decision": decision,
        "failed_checks": [name for name, value in checks.items() if not value],
        "maximum_absolute_errors": errors,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
