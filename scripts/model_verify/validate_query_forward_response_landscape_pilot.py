#!/usr/bin/env python3
"""Independently validate the Query forward-response landscape pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


DEFAULT_ARTIFACT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_forward_response_landscape_pilot_20260902_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def hadamard() -> np.ndarray:
    value = np.ones((1, 1), np.float64)
    while len(value) < 16:
        value = np.block([[value, value], [value, -value]])
    return value


def dct() -> np.ndarray:
    n = np.arange(16, dtype=np.float64)
    k = np.arange(16, dtype=np.float64)[:, None]
    value = np.cos(np.pi * (n[None] + 0.5) * k / 16.0)
    value[0] /= 4.0
    value[1:] *= math.sqrt(2.0 / 16.0)
    return value * 4.0


def qr(seed: int) -> np.ndarray:
    random = np.random.default_rng(seed).standard_normal((16, 16))
    q, r = np.linalg.qr(random)
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None]
    return q.T * 4.0


def expected_bases() -> np.ndarray:
    return np.stack((hadamard(), dct(), qr(260902), qr(260903))).astype(np.float32)


def interpolate(knots: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(knots, dtype=torch.float32)
    return (
        F.interpolate(value.transpose(1, 2), 50, mode="linear", align_corners=True)
        .transpose(1, 2)
        .numpy()
    )


def residual(
    trajectory: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    current_action: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    target = reference[1:]
    yaw = trajectory[..., 2] - target[None, :, 2]
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    previous = np.concatenate(
        (np.tile(current_action[None, None], (len(actions), 1, 1)), actions[:, :-1]),
        axis=1,
    )
    rate = actions - previous
    return np.concatenate(
        (
            np.sqrt(weights["position"])
            * (trajectory[..., :2] - target[None, :, :2]),
            np.sqrt(weights["yaw"]) * yaw[..., None],
            np.sqrt(weights["vx"])
            * (trajectory[..., 3] - target[None, :, 3])[..., None],
            np.sqrt(weights["acceleration_rate"]) * rate[..., :1],
            np.sqrt(weights["steering_rate"]) * rate[..., 1:2],
        ),
        axis=2,
    ).reshape(len(actions), -1).astype(np.float32)


def query_evaluate(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    knots: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    actions = interpolate(knots)
    result = controller.evaluate_action_sequences(
        data["state"][row],
        data["current_action"][row],
        data["history"][row : row + 1],
        data["reference"][row],
        actions,
    )
    costs = result["cost"].cpu().numpy().astype(np.float32)
    trajectories = result["trajectories"].cpu().numpy().astype(np.float32)
    return costs, residual(
        trajectories, actions, data["reference"][row], data["current_action"][row], weights
    )


def solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(left, right, rcond=None)[0]


def unit_rms(vector: np.ndarray) -> np.ndarray:
    value = float(np.sqrt(np.mean(vector * vector)))
    return vector / value if value > 1e-10 else np.zeros_like(vector)


def independently_fit(
    center: np.ndarray,
    center_cost: float,
    center_residual: np.ndarray,
    probes: np.ndarray,
    probe_cost: np.ndarray,
    probe_residual: np.ndarray,
    sigma: np.ndarray,
    ridge: float,
    damping: float,
    cap: float,
) -> dict[str, np.ndarray]:
    x = ((probes - center) / sigma).reshape(32, 16).astype(np.float64)
    x = np.vstack((np.zeros((1, 16)), x))
    dc = np.r_[0.0, probe_cost.astype(np.float64) - center_cost]
    dr = np.vstack(
        (np.zeros((1, 300)), probe_residual.astype(np.float64) - center_residual)
    )
    scale = max(float(np.quantile(np.abs(dc), 0.5)), 0.25)
    weight = 1.0 / (1.0 + (np.abs(dc) / scale) ** 2)
    weight /= weight.mean()
    gram = x.T @ (weight[:, None] * x) + ridge * np.eye(16)
    gradient = solve(gram, x.T @ (weight * dc))
    response = solve(gram, x.T @ (weight[:, None] * dr))
    step = -solve(response @ response.T + damping * np.eye(16), response @ center_residual)
    rho = float(np.sqrt(np.mean(step * step)))
    if rho > cap:
        step *= cap / rho
    cost_direction = unit_rms(-gradient)
    trajectory_direction = unit_rms(step)
    if np.mean(cost_direction * trajectory_direction) < 0.0:
        trajectory_direction *= -1.0
    blend = unit_rms(cost_direction + trajectory_direction)
    return {
        "x": x.astype(np.float32),
        "weight": weight.astype(np.float32),
        "gradient": gradient.astype(np.float32),
        "response": response.astype(np.float32),
        "step": step.astype(np.float32),
        "directions": np.stack((cost_direction, trajectory_direction, blend)).astype(np.float32),
    }


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    manifest_path = artifact / "manifest.json"
    config_path = artifact / "config.json"
    summary_path = artifact / "summary.json"
    rows_path = artifact / "rows.csv"
    landscape_path = artifact / "landscape.npz"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    checks: dict[str, bool] = {}
    errors: dict[str, float | int | str] = {}

    checks["manifest_pending"] = manifest["qualification"] == "PENDING_INDEPENDENT_VALIDATION"
    checks["artifact_hashes"] = (
        sha256(config_path) == manifest["config_sha256"]
        and sha256(summary_path) == manifest["summary_sha256"]
        and sha256(rows_path) == manifest["rows_sha256"]
        and sha256(landscape_path) == manifest["landscape_sha256"]
    )
    source = Path(manifest["source_audit"])
    checks["source_hashes"] = (
        sha256(source / "manifest.json") == manifest["source_manifest_sha256"]
        and sha256(source / "validation.json") == manifest["source_validation_sha256"]
        and sha256(source / "audit.npz") == manifest["source_audit_sha256"]
    )
    source_validation = json.loads((source / "validation.json").read_text())
    checks["source_qualified"] = (
        source_validation["qualification"]
        == "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS"
    )
    checks["sealed_boundaries"] = (
        not manifest["formal_validation_or_test_consumed"]
        and not manifest["dbm_fields_or_labels_consumed"]
        and not summary["formal_validation_or_test_consumed"]
        and not summary["dbm_fields_or_labels_consumed"]
    )

    with np.load(landscape_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(source / "audit.npz", allow_pickle=False) as archive:
        rows = data["source_audit_row"]
        source_context_error = max(
            maximum_error(data[name], archive[name][rows])
            for name in ("state", "current_action", "history", "reference", "reference_ego")
        )
        source_metadata_match = all(
            np.array_equal(data[name], archive[name][rows])
            for name in (
                "episode_id",
                "speed_kph",
                "speed_index",
                "variant_index",
                "fold_id",
            )
        )
        comparator_error = max(
            maximum_error(data[target], archive[source_name][rows])
            for target, source_name in (
                ("old_warm_knots", "old_warm_knots"),
                ("old_warm_cost", "old_warm_cost"),
                ("old_t0_cost", "old_t0_cost"),
                ("old_fullrank_cost", "old_fullrank_cost"),
                ("prior_canonical_knots", "canonical_teacher_knots"),
                ("prior_canonical_cost", "canonical_teacher_cost"),
            )
        )
    errors["source_context_max_error"] = source_context_error
    errors["source_comparator_max_error"] = comparator_error
    checks["source_rows"] = source_context_error == 0.0 and comparator_error == 0.0 and source_metadata_match
    checks["selection_balance"] = (
        len(data["episode_id"]) == 20
        and len(np.unique(data["episode_id"])) == 20
        and [int(np.sum(data["fold_id"] == fold)) for fold in range(5)] == [4] * 5
        and "episode_028" in data["episode_id"]
    )

    bases = expected_bases()
    errors["basis_max_error"] = maximum_error(data["round_basis"], bases)
    checks["basis"] = errors["basis_max_error"] == 0.0 and all(
        np.linalg.matrix_rank(value) == 16 for value in bases
    )
    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    expected_initial = np.empty((20, 5, 8, 2), np.float32)
    for row in range(20):
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
    errors["initial_center_max_error"] = maximum_error(data["center_knots"][:, :, 0], expected_initial)
    checks["initial_centers"] = errors["initial_center_max_error"] == 0.0

    maxima = {
        "probe_raw": 0.0,
        "probe_clipped": 0.0,
        "fit_x": 0.0,
        "fit_weight": 0.0,
        "gradient": 0.0,
        "response": 0.0,
        "gn_step": 0.0,
        "directions": 0.0,
        "proposal_delta": 0.0,
        "proposal_raw": 0.0,
        "proposal_clipped": 0.0,
        "selection_cost": 0.0,
        "selection_knots": 0.0,
        "selection_residual": 0.0,
    }
    selection_codes_match = True
    ridge = float(config["response_contract"]["fit_ridge"])
    damping = float(config["response_contract"]["gauss_newton_damping"])
    line = np.asarray(config["response_contract"]["line_factors"], np.float32)
    radii = np.asarray(config["probe_contract"]["radius_sigma_by_round"], np.float32)
    caps = np.asarray(config["response_contract"]["maximum_step_sigma_by_round"], np.float32)
    for row in range(20):
        for branch in range(5):
            for round_index in range(4):
                center = data["center_knots"][row, branch, round_index]
                raw = np.stack(
                    [
                        center + sign * radii[round_index] * direction.reshape(8, 2) * sigma
                        for direction in bases[round_index]
                        for sign in (1.0, -1.0)
                    ]
                ).astype(np.float32)
                clipped = np.clip(raw, low, high)
                maxima["probe_raw"] = max(maxima["probe_raw"], maximum_error(raw, data["probe_raw_knots"][row, branch, round_index]))
                maxima["probe_clipped"] = max(maxima["probe_clipped"], maximum_error(clipped, data["probe_knots"][row, branch, round_index]))
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
                for output_name, saved_name, key in (
                    ("fit_x", "fit_x", "x"),
                    ("fit_weight", "fit_weight", "weight"),
                    ("gradient", "cost_gradient", "gradient"),
                    ("response", "trajectory_response", "response"),
                    ("gn_step", "gauss_newton_step_sigma", "step"),
                    ("directions", "response_direction", "directions"),
                ):
                    maxima[output_name] = max(
                        maxima[output_name],
                        maximum_error(fitted[key], data[saved_name][row, branch, round_index]),
                    )
                steps = []
                for direction_index in range(3):
                    base_step = fitted["step"] if direction_index == 1 else caps[round_index] * fitted["directions"][direction_index]
                    for factor in line:
                        steps.append(factor * base_step)
                steps = np.asarray(steps, np.float32)
                proposal_raw = center[None] + steps.reshape(6, 8, 2) * sigma
                proposal_clipped = np.clip(proposal_raw, low, high).astype(np.float32)
                maxima["proposal_delta"] = max(maxima["proposal_delta"], maximum_error(steps, data["proposal_delta_sigma"][row, branch, round_index]))
                maxima["proposal_raw"] = max(maxima["proposal_raw"], maximum_error(proposal_raw, data["proposal_raw_knots"][row, branch, round_index]))
                maxima["proposal_clipped"] = max(maxima["proposal_clipped"], maximum_error(proposal_clipped, data["proposal_knots"][row, branch, round_index]))
                costs = np.r_[
                    data["center_cost"][row, branch, round_index],
                    data["probe_cost"][row, branch, round_index],
                    data["proposal_cost"][row, branch, round_index],
                ]
                winner = int(np.argmin(costs))
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
                selection_codes_match &= (
                    code == int(data["selected_source"][row, branch, round_index])
                    and index == int(data["selected_local_index"][row, branch, round_index])
                )
                maxima["selection_cost"] = max(maxima["selection_cost"], abs(float(costs[winner]) - float(data["center_cost"][row, branch, round_index + 1])))
                maxima["selection_knots"] = max(maxima["selection_knots"], maximum_error(knots, data["center_knots"][row, branch, round_index + 1]))
                maxima["selection_residual"] = max(maxima["selection_residual"], maximum_error(res, data["center_residual"][row, branch, round_index + 1]))
    errors.update({f"reconstruction_{key}_max_error": value for key, value in maxima.items()})
    checks["probe_geometry"] = maxima["probe_raw"] == 0.0 and maxima["probe_clipped"] == 0.0
    checks["response_fit"] = max(maxima[key] for key in ("fit_x", "fit_weight", "gradient", "response", "gn_step", "directions")) <= 2e-5
    checks["proposal_geometry"] = max(maxima[key] for key in ("proposal_delta", "proposal_raw", "proposal_clipped")) <= 2e-6
    checks["selection_chain"] = selection_codes_match and max(maxima[key] for key in ("selection_cost", "selection_knots", "selection_residual")) == 0.0

    canonical_roles = data["branch_role"] == "canonical"
    expected_branch = np.flatnonzero(canonical_roles)[
        np.argmin(data["center_cost"][:, canonical_roles, -1], axis=1)
    ]
    expected_cost = data["center_cost"][np.arange(20), expected_branch, -1]
    expected_knots = data["center_knots"][np.arange(20), expected_branch, -1]
    errors["canonical_cost_max_error"] = maximum_error(expected_cost, data["canonical_final_cost"])
    errors["canonical_knots_max_error"] = maximum_error(expected_knots, data["canonical_final_knots"])
    checks["canonical_excludes_shadow"] = (
        np.array_equal(expected_branch, data["canonical_final_branch"])
        and np.all(data["canonical_final_branch"] < 4)
        and errors["canonical_cost_max_error"] == 0.0
        and errors["canonical_knots_max_error"] == 0.0
    )

    finite_arrays = all(
        np.all(np.isfinite(value))
        for value in data.values()
        if np.issubdtype(value.dtype, np.number)
    )
    residual_cost_errors = []
    for cost_name, residual_name in (
        ("center_cost", "center_residual"),
        ("probe_cost", "probe_residual"),
        ("proposal_cost", "proposal_residual"),
    ):
        reconstructed = np.sum(data[residual_name].astype(np.float64) ** 2, axis=-1)
        residual_cost_errors.append(maximum_error(reconstructed, data[cost_name]))
    errors["stored_residual_cost_max_error"] = max(residual_cost_errors)
    checks["finite_and_residual_cost"] = finite_arrays and errors["stored_residual_cost_max_error"] <= 5e-3

    collection_manifest = json.loads((Path(manifest["source_collection"]) / "manifest.json").read_text())
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    model = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(TorchQueryRolloutBackend(model), params, device=str(device))
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    replay_rows = []
    for speed in (40, 55, 70, 85, 100):
        candidates = np.flatnonzero(data["speed_kph"] == speed)
        if speed == 40:
            candidates = candidates[data["episode_id"][candidates] == "episode_028"]
        replay_rows.append(int(candidates[0]))
    replay_cost_error = 0.0
    replay_residual_error = 0.0
    for row in replay_rows:
        cost, res = query_evaluate(controller, data, row, data["center_knots"][row, :, 0], weights)
        replay_cost_error = max(replay_cost_error, maximum_error(cost, data["center_cost"][row, :, 0]))
        replay_residual_error = max(replay_residual_error, maximum_error(res, data["center_residual"][row, :, 0]))
        for branch in range(5):
            for round_index in range(4):
                cost, res = query_evaluate(controller, data, row, data["probe_knots"][row, branch, round_index], weights)
                replay_cost_error = max(replay_cost_error, maximum_error(cost, data["probe_cost"][row, branch, round_index]))
                replay_residual_error = max(replay_residual_error, maximum_error(res, data["probe_residual"][row, branch, round_index]))
                cost, res = query_evaluate(controller, data, row, data["proposal_knots"][row, branch, round_index], weights)
                replay_cost_error = max(replay_cost_error, maximum_error(cost, data["proposal_cost"][row, branch, round_index]))
                replay_residual_error = max(replay_residual_error, maximum_error(res, data["proposal_residual"][row, branch, round_index]))
    errors["query_replay_cost_max_error"] = replay_cost_error
    errors["query_replay_residual_max_error"] = replay_residual_error
    checks["representative_query_replay"] = replay_cost_error <= 1e-5 and replay_residual_error <= 1e-5
    errors["query_replay_rows"] = ",".join(str(value) for value in replay_rows)
    errors["query_replay_rollouts"] = len(replay_rows) * 765

    warm = data["old_warm_cost"].astype(np.float64)
    prior = data["prior_canonical_cost"].astype(np.float64)
    final = data["canonical_final_cost"].astype(np.float64)
    first_two = data["response_incremental_gain"][:, canonical_roles, :2] > 1e-6
    trajectory_improves = (
        np.min(data["proposal_cost"][..., 2:4], axis=-1)
        < data["probe_only_best_cost"] - 1e-6
    )
    trajectory_fraction = float(np.mean(trajectory_improves[:, canonical_roles]))
    first_two_fraction = float(np.mean(first_two))
    aggregate_fraction = float(np.sum(prior - final) / np.sum(prior))
    speed_gains = {
        str(speed): float(np.mean(prior[data["speed_kph"] == speed] - final[data["speed_kph"] == speed]))
        for speed in (40, 55, 70, 85, 100)
    }
    gate_config = config["pre_registered_expand_to_100_state_gates"]
    gates = {
        "canonical_final_mean_cost_no_greater_than_old_warm": bool(final.mean() <= warm.mean() + 1e-8),
        "canonical_final_warm_regression_fraction_le_0p25": bool(np.mean(final > warm + 1e-5) <= gate_config["canonical_final_warm_regression_fraction_maximum"]),
        "canonical_final_aggregate_gain_vs_prior_canonical_ge_0p10": bool(aggregate_fraction >= gate_config["canonical_final_aggregate_gain_vs_prior_canonical_minimum_fraction"]),
        "first_two_round_response_proposal_improvement_fraction_ge_0p50": bool(first_two_fraction >= gate_config["first_two_round_response_proposal_improvement_fraction_minimum"]),
        "trajectory_response_realized_improvement_positive_fraction_ge_0p50": bool(trajectory_fraction >= gate_config["trajectory_response_realized_improvement_positive_fraction_minimum"]),
        "all_speed_groups_nonnegative_mean_gain_vs_prior_canonical": bool(all(value >= -1e-8 for value in speed_gains.values())),
    }
    cost_pass = all(
        gates[name]
        for name in (
            "canonical_final_mean_cost_no_greater_than_old_warm",
            "canonical_final_warm_regression_fraction_le_0p25",
            "canonical_final_aggregate_gain_vs_prior_canonical_ge_0p10",
            "all_speed_groups_nonnegative_mean_gain_vs_prior_canonical",
        )
    )
    response_pass = all(
        gates[name]
        for name in (
            "first_two_round_response_proposal_improvement_fraction_ge_0p50",
            "trajectory_response_realized_improvement_positive_fraction_ge_0p50",
        )
    )
    if cost_pass and response_pass:
        expected_decision = "EXPAND_TO_REMAINING_80_PENDING_INDEPENDENT_REPLAY"
    elif response_pass:
        expected_decision = "RESPONSE_WORKS_COST_FAIL_ADJUST_GEOMETRY_ON_SAME_20"
    else:
        expected_decision = "RESPONSE_CONSTRUCTION_FAIL_STOP_NO_LARGE_ONESHOT_BANK"
    errors["first_two_response_fraction"] = first_two_fraction
    errors["trajectory_response_fraction"] = trajectory_fraction
    errors["aggregate_gain_vs_prior_fraction"] = aggregate_fraction
    checks["metrics_and_decision"] = (
        gates == summary["pre_registered_performance_gates"]
        and expected_decision == summary["performance_decision"]
        and expected_decision == manifest["performance_decision"]
    )
    checks["budget"] = (
        summary["rollouts_per_state"] == 765
        and summary["total_new_query_rollouts"] == 15300
        and manifest["total_new_query_rollouts"] == 15300
    )

    qualification = (
        "QUERY_FORWARD_RESPONSE_LANDSCAPE_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_FORWARD_RESPONSE_LANDSCAPE_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "performance_decision": expected_decision,
        "checks": checks,
        "errors_and_metrics": errors,
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
