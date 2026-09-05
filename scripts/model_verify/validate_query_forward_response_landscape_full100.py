#!/usr/bin/env python3
"""Independently validate the incremental full-100 Query landscape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)
from run_query_forward_response_landscape_pilot import sha256  # noqa: E402
from validate_query_forward_response_landscape_pilot import (  # noqa: E402
    expected_bases,
    independently_fit,
    maximum_error,
    query_evaluate,
)


DEFAULT_ARTIFACT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_forward_response_landscape_full100_20260902_v1"
)
SHARED_FIELDS = (
    "branch_name",
    "branch_role",
    "round_basis",
    "round_radius_sigma",
    "round_maximum_step_sigma",
    "line_factor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    manifest_path = artifact / "manifest.json"
    config_path = artifact / "config.json"
    search_path = artifact / "search_contract.json"
    summary_path = artifact / "summary.json"
    rows_path = artifact / "rows.csv"
    landscape_path = artifact / "landscape.npz"
    manifest = json.loads(manifest_path.read_text())
    expand_config = json.loads(config_path.read_text())
    search = json.loads(search_path.read_text())
    summary = json.loads(summary_path.read_text())
    checks: dict[str, bool] = {}
    metrics: dict[str, float | int | str] = {}

    checks["artifact_hashes"] = (
        sha256(config_path) == manifest["config_sha256"]
        and sha256(search_path) == manifest["search_contract_sha256"]
        and sha256(summary_path) == manifest["summary_sha256"]
        and sha256(rows_path) == manifest["rows_sha256"]
        and sha256(landscape_path) == manifest["landscape_sha256"]
    )
    source = Path(manifest["source_audit"])
    pilot_path = Path(manifest["embedded_pilot"])
    checks["source_hashes"] = (
        sha256(source / "manifest.json") == manifest["source_manifest_sha256"]
        and sha256(source / "validation.json") == manifest["source_validation_sha256"]
        and sha256(source / "audit.npz") == manifest["source_audit_sha256"]
    )
    checks["pilot_hashes"] = (
        sha256(pilot_path / "manifest.json")
        == manifest["embedded_pilot_manifest_sha256"]
        and sha256(pilot_path / "validation.json")
        == manifest["embedded_pilot_validation_sha256"]
        and sha256(pilot_path / "landscape.npz")
        == manifest["embedded_pilot_landscape_sha256"]
    )
    source_validation = json.loads((source / "validation.json").read_text())
    pilot_validation = json.loads((pilot_path / "validation.json").read_text())
    checks["parent_qualifications"] = (
        source_validation["qualification"]
        == "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS"
        and pilot_validation["qualification"]
        == "QUERY_FORWARD_RESPONSE_LANDSCAPE_INDEPENDENT_PASS"
    )
    checks["sealed_boundaries"] = (
        not manifest["formal_validation_or_test_consumed"]
        and not manifest["dbm_fields_or_labels_consumed"]
        and not summary["formal_validation_or_test_consumed"]
        and not summary["dbm_fields_or_labels_consumed"]
        and not expand_config["formal_validation_or_test_consumed"]
        and not expand_config["dbm_fields_or_labels_consumed"]
    )

    with np.load(landscape_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(pilot_path / "landscape.npz", allow_pickle=False) as archive:
        pilot = {name: np.asarray(archive[name]) for name in archive.files}
    pilot_error = 0.0
    pilot_exact = True
    pilot_rows = pilot["source_audit_row"]
    for name in pilot:
        if name in SHARED_FIELDS:
            left = data[name]
        else:
            left = data[name][pilot_rows]
        if np.issubdtype(left.dtype, np.number):
            pilot_error = max(pilot_error, maximum_error(left, pilot[name]))
        else:
            pilot_exact &= np.array_equal(left, pilot[name])
    metrics["embedded_pilot_max_error"] = pilot_error
    checks["embedded_pilot_exact"] = pilot_error == 0.0 and pilot_exact

    with np.load(source / "audit.npz", allow_pickle=False) as archive:
        source_numeric_error = max(
            maximum_error(data[name], archive[name])
            for name in (
                "state",
                "current_action",
                "history",
                "reference",
                "reference_ego",
            )
        )
        source_meta = all(
            np.array_equal(data[name], archive[name])
            for name in (
                "episode_id",
                "speed_kph",
                "speed_index",
                "variant_index",
                "fold_id",
            )
        )
        comparator_error = max(
            maximum_error(data[target], archive[source_name])
            for target, source_name in (
                ("old_warm_knots", "old_warm_knots"),
                ("old_warm_cost", "old_warm_cost"),
                ("old_t0_cost", "old_t0_cost"),
                ("old_fullrank_cost", "old_fullrank_cost"),
                ("prior_canonical_knots", "canonical_teacher_knots"),
                ("prior_canonical_cost", "canonical_teacher_cost"),
            )
        )
    metrics["source_numeric_max_error"] = source_numeric_error
    metrics["source_comparator_max_error"] = comparator_error
    checks["source_consolidation"] = (
        source_numeric_error == 0.0
        and comparator_error == 0.0
        and source_meta
        and np.array_equal(data["source_audit_row"], np.arange(100))
    )
    checks["coverage_balance"] = (
        len(np.unique(data["episode_id"])) == 100
        and [int(np.sum(data["fold_id"] == fold)) for fold in range(5)] == [20] * 5
        and [int(np.sum(data["speed_kph"] == speed)) for speed in (40, 55, 70, 85, 100)]
        == [20] * 5
    )

    bases = expected_bases()
    metrics["basis_max_error"] = maximum_error(data["round_basis"], bases)
    checks["basis"] = metrics["basis_max_error"] == 0.0 and all(
        np.linalg.matrix_rank(value) == 16 for value in bases
    )
    sigma = np.asarray(search["noise_sigma"], np.float32)
    low = np.asarray(search["action_bounds"]["minimum"], np.float32)
    high = np.asarray(search["action_bounds"]["maximum"], np.float32)
    expected_initial = np.empty((100, 5, 8, 2), np.float32)
    for row in range(100):
        action = data["current_action"][row]
        expected_initial[row] = np.stack(
            (
                np.broadcast_to(action, (8, 2)),
                np.linspace(action, np.zeros(2, np.float32), 8),
                np.zeros((8, 2), np.float32),
                data["prior_canonical_knots"][row],
                data["old_warm_knots"][row],
            )
        )
    expected_initial = np.clip(expected_initial, low, high)
    metrics["initial_center_max_error"] = maximum_error(
        data["center_knots"][:, :, 0], expected_initial
    )
    checks["initial_centers"] = metrics["initial_center_max_error"] == 0.0

    maxima = {
        "probe": 0.0,
        "fit": 0.0,
        "proposal": 0.0,
        "selection": 0.0,
    }
    selection_exact = True
    ridge = float(search["response_contract"]["fit_ridge"])
    damping = float(search["response_contract"]["gauss_newton_damping"])
    line = np.asarray(search["response_contract"]["line_factors"], np.float32)
    radii = np.asarray(search["probe_contract"]["radius_sigma_by_round"], np.float32)
    caps = np.asarray(search["response_contract"]["maximum_step_sigma_by_round"], np.float32)
    for row in range(100):
        for branch in range(5):
            for round_index in range(4):
                center = data["center_knots"][row, branch, round_index]
                raw = np.stack(
                    [
                        center
                        + sign
                        * radii[round_index]
                        * direction.reshape(8, 2)
                        * sigma
                        for direction in bases[round_index]
                        for sign in (1.0, -1.0)
                    ]
                ).astype(np.float32)
                clipped = np.clip(raw, low, high)
                maxima["probe"] = max(
                    maxima["probe"],
                    maximum_error(raw, data["probe_raw_knots"][row, branch, round_index]),
                    maximum_error(clipped, data["probe_knots"][row, branch, round_index]),
                )
                fitted = independently_fit(
                    center,
                    float(data["center_cost"][row, branch, round_index]),
                    data["center_residual"][row, branch, round_index],
                    clipped,
                    data["probe_cost"][row, branch, round_index],
                    data["probe_residual"][row, branch, round_index],
                    sigma,
                    ridge,
                    damping,
                    float(caps[round_index]),
                )
                for saved_name, fit_name in (
                    ("fit_x", "x"),
                    ("fit_weight", "weight"),
                    ("cost_gradient", "gradient"),
                    ("trajectory_response", "response"),
                    ("gauss_newton_step_sigma", "step"),
                    ("response_direction", "directions"),
                ):
                    maxima["fit"] = max(
                        maxima["fit"],
                        maximum_error(
                            fitted[fit_name], data[saved_name][row, branch, round_index]
                        ),
                    )
                steps = []
                for direction_index in range(3):
                    base = (
                        fitted["step"]
                        if direction_index == 1
                        else caps[round_index] * fitted["directions"][direction_index]
                    )
                    for factor in line:
                        steps.append(factor * base)
                steps = np.asarray(steps, np.float32)
                proposal_raw = center[None] + steps.reshape(6, 8, 2) * sigma
                proposal = np.clip(proposal_raw, low, high).astype(np.float32)
                maxima["proposal"] = max(
                    maxima["proposal"],
                    maximum_error(
                        steps, data["proposal_delta_sigma"][row, branch, round_index]
                    ),
                    maximum_error(
                        proposal_raw,
                        data["proposal_raw_knots"][row, branch, round_index],
                    ),
                    maximum_error(
                        proposal, data["proposal_knots"][row, branch, round_index]
                    ),
                )
                all_cost = np.r_[
                    data["center_cost"][row, branch, round_index],
                    data["probe_cost"][row, branch, round_index],
                    data["proposal_cost"][row, branch, round_index],
                ]
                winner = int(np.argmin(all_cost))
                if winner == 0:
                    code, index = 0, 0
                    knots = center
                    res = data["center_residual"][row, branch, round_index]
                elif winner <= 32:
                    code, index = 1, winner - 1
                    knots = data["probe_knots"][row, branch, round_index, index]
                    res = data["probe_residual"][row, branch, round_index, index]
                else:
                    code, index = 2, winner - 33
                    knots = data["proposal_knots"][row, branch, round_index, index]
                    res = data["proposal_residual"][row, branch, round_index, index]
                selection_exact &= (
                    code == int(data["selected_source"][row, branch, round_index])
                    and index == int(data["selected_local_index"][row, branch, round_index])
                )
                maxima["selection"] = max(
                    maxima["selection"],
                    abs(
                        float(all_cost[winner])
                        - float(data["center_cost"][row, branch, round_index + 1])
                    ),
                    maximum_error(
                        knots, data["center_knots"][row, branch, round_index + 1]
                    ),
                    maximum_error(
                        res, data["center_residual"][row, branch, round_index + 1]
                    ),
                )
    metrics.update({f"reconstruction_{key}_max_error": value for key, value in maxima.items()})
    checks["probe_geometry"] = maxima["probe"] == 0.0
    checks["response_fit"] = maxima["fit"] <= 2e-5
    checks["proposal_geometry"] = maxima["proposal"] <= 2e-6
    checks["selection_chain"] = selection_exact and maxima["selection"] == 0.0

    canonical = data["branch_role"] == "canonical"
    expected_branch = np.flatnonzero(canonical)[
        np.argmin(data["center_cost"][:, canonical, -1], axis=1)
    ]
    rows = np.arange(100)
    expected_cost = data["center_cost"][rows, expected_branch, -1]
    expected_knots = data["center_knots"][rows, expected_branch, -1]
    metrics["canonical_cost_max_error"] = maximum_error(
        expected_cost, data["canonical_final_cost"]
    )
    metrics["canonical_knots_max_error"] = maximum_error(
        expected_knots, data["canonical_final_knots"]
    )
    checks["canonical_excludes_shadow"] = (
        np.array_equal(expected_branch, data["canonical_final_branch"])
        and np.all(expected_branch < 4)
        and metrics["canonical_cost_max_error"] == 0.0
        and metrics["canonical_knots_max_error"] == 0.0
    )
    residual_error = 0.0
    for cost_name, residual_name in (
        ("center_cost", "center_residual"),
        ("probe_cost", "probe_residual"),
        ("proposal_cost", "proposal_residual"),
    ):
        residual_error = max(
            residual_error,
            maximum_error(
                np.sum(data[residual_name].astype(np.float64) ** 2, axis=-1),
                data[cost_name],
            ),
        )
    metrics["stored_residual_cost_max_error"] = residual_error
    checks["finite_and_residual_cost"] = (
        all(
            np.all(np.isfinite(value))
            for value in data.values()
            if np.issubdtype(value.dtype, np.number)
        )
        and residual_error <= 5e-3
    )

    collection_manifest = json.loads(
        (Path(manifest["source_collection"]) / "manifest.json").read_text()
    )
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    model = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(model), params, device=str(device)
    )
    weights = {name: float(value) for name, value in search["cost_weights"].items()}
    remaining = np.setdiff1d(np.arange(100), pilot_rows)
    replay_rows = []
    for speed in (40, 55, 70, 85, 100):
        for fold in (0, 2):
            candidates = remaining[
                (data["speed_kph"][remaining] == speed)
                & (data["fold_id"][remaining] == fold)
            ]
            replay_rows.append(int(candidates[0]))
    replay_cost_error = 0.0
    replay_residual_error = 0.0
    for row in replay_rows:
        cost, res = query_evaluate(
            controller, data, row, data["center_knots"][row, :, 0], weights
        )
        replay_cost_error = max(
            replay_cost_error, maximum_error(cost, data["center_cost"][row, :, 0])
        )
        replay_residual_error = max(
            replay_residual_error,
            maximum_error(res, data["center_residual"][row, :, 0]),
        )
        for branch in range(5):
            for round_index in range(4):
                cost, res = query_evaluate(
                    controller,
                    data,
                    row,
                    data["probe_knots"][row, branch, round_index],
                    weights,
                )
                replay_cost_error = max(
                    replay_cost_error,
                    maximum_error(cost, data["probe_cost"][row, branch, round_index]),
                )
                replay_residual_error = max(
                    replay_residual_error,
                    maximum_error(
                        res, data["probe_residual"][row, branch, round_index]
                    ),
                )
                cost, res = query_evaluate(
                    controller,
                    data,
                    row,
                    data["proposal_knots"][row, branch, round_index],
                    weights,
                )
                replay_cost_error = max(
                    replay_cost_error,
                    maximum_error(
                        cost, data["proposal_cost"][row, branch, round_index]
                    ),
                )
                replay_residual_error = max(
                    replay_residual_error,
                    maximum_error(
                        res, data["proposal_residual"][row, branch, round_index]
                    ),
                )
    metrics["query_replay_rows"] = ",".join(str(value) for value in replay_rows)
    metrics["query_replay_rollouts"] = len(replay_rows) * 765
    metrics["query_replay_cost_max_error"] = replay_cost_error
    metrics["query_replay_residual_max_error"] = replay_residual_error
    checks["new_row_query_replay"] = (
        replay_cost_error <= 1e-5 and replay_residual_error <= 1e-5
    )

    final = data["canonical_final_cost"].astype(np.float64)
    warm = data["old_warm_cost"].astype(np.float64)
    prior = data["prior_canonical_cost"].astype(np.float64)
    first_two_fraction = float(
        np.mean(data["response_incremental_gain"][:, canonical, :2] > 1e-6)
    )
    trajectory_fraction = float(
        np.mean(
            np.min(data["proposal_cost"][:, canonical, :, 2:4], axis=-1)
            < data["probe_only_best_cost"][:, canonical] - 1e-6
        )
    )
    aggregate_fraction = float(np.sum(prior - final) / np.sum(prior))
    speed_gains = {
        str(speed): float(
            np.mean(prior[data["speed_kph"] == speed] - final[data["speed_kph"] == speed])
        )
        for speed in (40, 55, 70, 85, 100)
    }
    gate_config = search["pre_registered_expand_to_100_state_gates"]
    gates = {
        "canonical_final_mean_cost_no_greater_than_old_warm": bool(
            final.mean() <= warm.mean() + 1e-8
        ),
        "canonical_final_warm_regression_fraction_le_0p25": bool(
            np.mean(final > warm + 1e-5)
            <= gate_config["canonical_final_warm_regression_fraction_maximum"]
        ),
        "canonical_final_aggregate_gain_vs_prior_canonical_ge_0p10": bool(
            aggregate_fraction
            >= gate_config[
                "canonical_final_aggregate_gain_vs_prior_canonical_minimum_fraction"
            ]
        ),
        "first_two_round_response_proposal_improvement_fraction_ge_0p50": bool(
            first_two_fraction
            >= gate_config[
                "first_two_round_response_proposal_improvement_fraction_minimum"
            ]
        ),
        "trajectory_response_realized_improvement_positive_fraction_ge_0p50": bool(
            trajectory_fraction
            >= gate_config[
                "trajectory_response_realized_improvement_positive_fraction_minimum"
            ]
        ),
        "all_speed_groups_nonnegative_mean_gain_vs_prior_canonical": bool(
            all(value >= -1e-8 for value in speed_gains.values())
        ),
    }
    if all(gates.values()):
        expected_decision = (
            "FULL100_LANDSCAPE_READY_FOR_ABSOLUTE_CRITIC_REPLAY_PENDING_VALIDATION"
        )
    elif (
        gates["first_two_round_response_proposal_improvement_fraction_ge_0p50"]
        and gates[
            "trajectory_response_realized_improvement_positive_fraction_ge_0p50"
        ]
    ):
        expected_decision = "FULL100_RESPONSE_WORKS_COST_GATE_FAIL_NO_CRITIC_REPLAY"
    else:
        expected_decision = "FULL100_RESPONSE_FAIL_STOP"
    metrics["first_two_response_fraction"] = first_two_fraction
    metrics["trajectory_response_fraction"] = trajectory_fraction
    metrics["aggregate_gain_vs_prior_fraction"] = aggregate_fraction
    checks["metrics_and_decision"] = (
        gates == summary["pre_registered_performance_gates"]
        and expected_decision == summary["performance_decision"]
        and expected_decision == manifest["performance_decision"]
    )
    checks["budget"] = (
        summary["new_query_rollouts"] == 61200
        and summary["represented_query_rollouts"] == 76500
        and manifest["new_query_rollouts"] == 61200
        and manifest["represented_query_rollouts"] == 76500
    )

    qualification = (
        "QUERY_FORWARD_RESPONSE_FULL100_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_FORWARD_RESPONSE_FULL100_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "performance_decision": (
            "FULL100_LANDSCAPE_READY_FOR_ABSOLUTE_CRITIC_REPLAY"
            if qualification.endswith("PASS") and all(gates.values())
            else expected_decision
        ),
        "checks": checks,
        "metrics": metrics,
        "pre_registered_performance_gates": gates,
        "speed_mean_gain_vs_prior_canonical": speed_gains,
        "artifact": str(artifact),
        "manifest_sha256": sha256(manifest_path),
        "landscape_sha256": sha256(landscape_path),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }
    dump_json(artifact / "validation.json", validation)
    print(json.dumps(validation, indent=2))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
