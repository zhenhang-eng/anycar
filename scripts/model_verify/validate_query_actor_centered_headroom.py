#!/usr/bin/env python3
"""Independently replay and validate the Actor-centered Query headroom audit."""

from __future__ import annotations

import argparse
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

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_actor_centered_headroom import headroom_metrics, pooled_metrics, subset  # noqa: E402
from run_query_forward_response_landscape_pilot import (  # noqa: E402
    basis_bank,
    evaluate,
    fit_response,
    sha256,
)
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    load_inputs,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_actor_centered_headroom_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def update_error(errors: dict[str, float], name: str, left: np.ndarray, right: np.ndarray) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise AssertionError(f"shape mismatch for {name}: {left.shape} != {right.shape}")
    if left.dtype.kind in "OUSb" or right.dtype.kind in "OUSb":
        value = 0.0 if np.array_equal(left, right) else 1.0
    else:
        value = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    errors[name] = max(errors.get(name, 0.0), value)


def numeric_leaf_error(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("metric dictionary keys differ")
        return max((numeric_leaf_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise AssertionError(f"nonnumeric metric differs: {left!r} != {right!r}")
    return 0.0


def validate_seed(
    record: dict[str, Any],
    config: dict,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    controller: TorchMPPIController,
    device: torch.device,
) -> dict[str, Any]:
    errors: dict[str, float] = {}
    result_path = Path(record["arrays"])
    if sha256(result_path) != record["arrays_sha256"]:
        raise AssertionError("result array hash mismatch")
    checkpoint_path = Path(record["source_checkpoint"])
    source_arrays_path = Path(record["source_arrays"])
    if sha256(checkpoint_path) != record["source_checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    if sha256(source_arrays_path) != record["source_arrays_sha256"]:
        raise AssertionError("source OAC arrays hash mismatch")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    actor_inputs = load_inputs(data, payload["actor_normalization"])
    actor_knots = actor_predict(actor, actor_inputs, rows, device)
    local = subset(data, rows)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    radii = np.asarray(config["probe_contract"]["radius_sigma_by_round"], np.float32)
    maximum_steps = np.asarray(
        config["response_contract"]["maximum_step_sigma_by_round"], np.float32
    )
    line_factors = np.asarray(config["response_contract"]["line_factors"], np.float32)
    ridge = float(config["response_contract"]["fit_ridge"])
    damping = float(config["response_contract"]["gauss_newton_damping"])
    bases = basis_bank()
    rounds = int(config["round_count"])

    with np.load(source_arrays_path, allow_pickle=False) as source_arrays:
        update_error(errors, "source_selection_indices", source_arrays["selection_indices"], rows)
        source_action = np.asarray(
            source_arrays["selection_round_action"][int(payload["selected_round"])], np.float32
        )
        source_cost = np.asarray(
            source_arrays["selection_round_cost"][int(payload["selected_round"])], np.float32
        )
    with np.load(result_path, allow_pickle=False) as result:
        expected_shapes = {
            "centers": (len(rows), rounds + 1, 8, 2),
            "center_cost": (len(rows), rounds + 1),
            "probe_knots": (len(rows), rounds, 32, 8, 2),
            "probe_cost": (len(rows), rounds, 32),
            "proposal_knots": (len(rows), rounds, 6, 8, 2),
            "proposal_cost": (len(rows), rounds, 6),
        }
        for name, shape in expected_shapes.items():
            if result[name].shape != shape:
                raise AssertionError(f"unexpected {name} shape {result[name].shape}")
        update_error(errors, "selection_indices", result["selection_indices"], rows)
        update_error(errors, "episode_id", result["episode_id"], local["episode_id"])
        update_error(errors, "speed_kph", result["speed_kph"], local["speed_kph"])
        update_error(errors, "variant_index", result["variant_index"], local["variant_index"])
        update_error(errors, "actor_reload", result["actor_knots"], actor_knots)
        update_error(errors, "actor_source_action", result["actor_knots"], source_action)
        update_error(errors, "actor_center", result["centers"][:, 0], actor_knots)
        update_error(errors, "actor_source_cost", result["actor_source_cost"], source_cost)

        for row in range(len(rows)):
            replay_cost, replay_residual = evaluate(
                controller, local, row, actor_knots[row : row + 1], weights
            )
            update_error(errors, "actor_query_cost", result["center_cost"][row, 0:1], replay_cost)
            update_error(
                errors, "actor_query_residual",
                result["center_residual"][row, 0:1], replay_residual,
            )
            for round_index in range(rounds):
                incumbent = np.asarray(result["centers"][row, round_index], np.float32)
                incumbent_cost = float(result["center_cost"][row, round_index])
                incumbent_residual = np.asarray(
                    result["center_residual"][row, round_index], np.float32
                )
                basis = bases[round_index].reshape(16, 8, 2)
                expected_raw = np.stack([
                    incumbent + sign * float(radii[round_index]) * direction * sigma[None]
                    for direction in basis for sign in (1.0, -1.0)
                ]).astype(np.float32)
                expected_probes = np.clip(expected_raw, low, high).astype(np.float32)
                expected_clipped = np.abs(expected_raw - expected_probes) > 1e-7
                update_error(errors, "probe_raw", result["probe_raw"][row, round_index], expected_raw)
                update_error(errors, "probe_knots", result["probe_knots"][row, round_index], expected_probes)
                update_error(errors, "probe_clipped", result["probe_clipped"][row, round_index], expected_clipped)
                probe_cost, probe_residual = evaluate(
                    controller, local, row, expected_probes, weights
                )
                update_error(errors, "probe_query_cost", result["probe_cost"][row, round_index], probe_cost)
                update_error(
                    errors, "probe_query_residual",
                    result["probe_residual"][row, round_index], probe_residual,
                )
                fitted = fit_response(
                    incumbent, incumbent_cost, incumbent_residual,
                    expected_probes, probe_cost, probe_residual, sigma,
                    ridge, damping, float(maximum_steps[round_index]),
                )
                expected_directions = np.stack((
                    fitted["cost_direction"],
                    fitted["trajectory_direction"],
                    fitted["blend_direction"],
                ))
                update_error(errors, "cost_gradient", result["cost_gradient"][row, round_index], fitted["gradient"])
                update_error(errors, "trajectory_response", result["trajectory_response"][row, round_index], fitted["response"])
                update_error(errors, "gn_step", result["gn_step"][row, round_index], fitted["gn_step"])
                update_error(errors, "response_directions", result["response_directions"][row, round_index], expected_directions)
                update_error(errors, "cost_fit_error", result["cost_fit_error"][row, round_index], fitted["cost_fit_error"])
                update_error(errors, "residual_fit_error", result["residual_fit_error"][row, round_index], fitted["residual_fit_error"])
                steps = []
                for direction_index in range(3):
                    base_step = (
                        fitted["gn_step"] if direction_index == 1
                        else float(maximum_steps[round_index]) * expected_directions[direction_index]
                    )
                    for factor in line_factors:
                        steps.append(float(factor) * base_step)
                steps = np.stack(steps).astype(np.float32)
                expected_raw_proposals = incumbent[None] + steps.reshape(-1, 8, 2) * sigma[None, None]
                expected_proposals = np.clip(expected_raw_proposals, low, high).astype(np.float32)
                expected_proposal_clipped = np.abs(expected_raw_proposals - expected_proposals) > 1e-7
                update_error(errors, "proposal_steps", result["proposal_delta_sigma"][row, round_index], steps)
                update_error(errors, "proposal_raw", result["proposal_raw"][row, round_index], expected_raw_proposals)
                update_error(errors, "proposal_knots", result["proposal_knots"][row, round_index], expected_proposals)
                update_error(errors, "proposal_clipped", result["proposal_clipped"][row, round_index], expected_proposal_clipped)
                proposal_cost, proposal_residual = evaluate(
                    controller, local, row, expected_proposals, weights
                )
                update_error(errors, "proposal_query_cost", result["proposal_cost"][row, round_index], proposal_cost)
                update_error(
                    errors, "proposal_query_residual",
                    result["proposal_residual"][row, round_index], proposal_residual,
                )
                combined = np.concatenate(([incumbent_cost], probe_cost, proposal_cost))
                winner = int(np.argmin(combined))
                if winner == 0:
                    expected_source, expected_index = 0, 0
                    next_knots, next_residual = incumbent, incumbent_residual
                elif winner <= 32:
                    expected_source, expected_index = 1, winner - 1
                    next_knots, next_residual = expected_probes[expected_index], probe_residual[expected_index]
                else:
                    expected_source, expected_index = 2, winner - 33
                    next_knots, next_residual = expected_proposals[expected_index], proposal_residual[expected_index]
                update_error(errors, "selected_source", result["selected_source"][row, round_index], expected_source)
                update_error(errors, "selected_local_index", result["selected_local_index"][row, round_index], expected_index)
                update_error(errors, "next_center", result["centers"][row, round_index + 1], next_knots)
                update_error(errors, "next_cost", result["center_cost"][row, round_index + 1], combined[winner])
                update_error(errors, "next_residual", result["center_residual"][row, round_index + 1], next_residual)
            if row == 0 or (row + 1) % 20 == 0:
                print(f"validate seed={record['seed']} {row + 1}/{len(rows)}", flush=True)

        recomputed_round_metrics = [
            headroom_metrics(
                result["center_cost"][:, 0], result["center_cost"][:, round_index],
                result["centers"][:, 0], result["centers"][:, round_index], sigma,
                result["speed_kph"], result["variant_index"],
            )
            for round_index in range(rounds + 1)
        ]
    metric_error = numeric_leaf_error(record["round_metrics"], recomputed_round_metrics)
    return {
        "seed": int(record["seed"]),
        "max_errors": errors,
        "round_metric_max_abs_error": metric_error,
        "all_array_and_query_checks_pass": all(value <= 1e-6 for value in errors.values()),
    }


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
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"], summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"], summary["actor_or_critic_trained"])):
        raise AssertionError("sealed boundary or no-training contract violated")

    source_validation = Path(config["sources"]["actor_oac"]) / "validation.json"
    if sha256(source_validation) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation.read_text())["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source Actor qualification changed")

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
        expected_decision = "QUERY_ACTOR_HAS_MATERIAL_HEADROOM_ADVANCE_SINGLE_VARIABLE_AB"
    checks = {
        "artifact_hashes": True,
        "source_actor_and_replay_qualified": True,
        "split_and_sealed_boundary": True,
        "expected_rollout_budget": int(config["query_rollout_budget"]["total_query_rollouts"]) == 55080,
        "actor_reload_exact": all(report["max_errors"].get("actor_reload", 1.0) == 0.0 for report in reports),
        "all_saved_query_costs_and_residuals_replay": all(report["all_array_and_query_checks_pass"] for report in reports),
        "response_fit_and_winner_chain_reconstruct": all(report["all_array_and_query_checks_pass"] for report in reports),
        "metrics_recompute": pooled_metric_error <= 1e-12 and all(report["round_metric_max_abs_error"] <= 1e-12 for report in reports),
        "decision_recompute": summary["decision"] == expected_decision,
    }
    passed = all(checks.values())
    report = {
        "qualification": (
            "QUERY_ACTOR_CENTERED_HEADROOM_INDEPENDENT_PASS"
            if passed else "QUERY_ACTOR_CENTERED_HEADROOM_INDEPENDENT_FAIL"
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
