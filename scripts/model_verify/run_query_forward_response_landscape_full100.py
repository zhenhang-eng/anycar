#!/usr/bin/env python3
"""Expand the validated 20-state Query landscape to all 100 audit states.

The old pilot is embedded byte-for-array without rerunning it.  Only the other
80 episode-independent states receive the frozen 765-rollout adaptive search.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
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
from run_query_forward_response_landscape_pilot import (  # noqa: E402
    CONTEXT_FIELDS,
    basin_count,
    basis_bank,
    evaluate,
    find_collection,
    fit_response,
    grouped_metrics,
    sha256,
    stats,
)


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_forward_response_full100_config_20260902_v1.json"
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_source_rows(source: Path, rows: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(source / "audit.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name][rows]) for name in CONTEXT_FIELDS}
        data.update(
            {
                "source_audit_row": rows.astype(np.int64),
                "old_warm_knots": np.asarray(archive["old_warm_knots"][rows]),
                "old_warm_cost": np.asarray(archive["old_warm_cost"][rows]),
                "old_t0_cost": np.asarray(archive["old_t0_cost"][rows]),
                "old_fullrank_cost": np.asarray(archive["old_fullrank_cost"][rows]),
                "prior_canonical_knots": np.asarray(
                    archive["canonical_teacher_knots"][rows]
                ),
                "prior_canonical_cost": np.asarray(
                    archive["canonical_teacher_cost"][rows]
                ),
            }
        )
    return data


def run_search(
    data: dict[str, np.ndarray],
    controller: TorchMPPIController,
    config: dict,
) -> dict[str, np.ndarray]:
    branch_names = np.asarray([value["name"] for value in config["branches"]])
    branch_roles = np.asarray([value["role"] for value in config["branches"]])
    canonical_branch = branch_roles == "canonical"
    branch_count = len(branch_names)
    round_count = int(config["round_count"])
    row_count = len(data["state"])
    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    bases = basis_bank()
    radii = np.asarray(config["probe_contract"]["radius_sigma_by_round"], np.float32)
    maximum_steps = np.asarray(
        config["response_contract"]["maximum_step_sigma_by_round"], np.float32
    )
    line_factors = np.asarray(config["response_contract"]["line_factors"], np.float32)
    ridge = float(config["response_contract"]["fit_ridge"])
    damping = float(config["response_contract"]["gauss_newton_damping"])
    probe_count, proposal_count, residual_dim = 32, 6, 300

    centers = np.empty((row_count, branch_count, round_count + 1, 8, 2), np.float32)
    center_cost = np.empty((row_count, branch_count, round_count + 1), np.float32)
    center_residual = np.empty(
        (row_count, branch_count, round_count + 1, residual_dim), np.float32
    )
    probe_raw = np.empty((row_count, branch_count, round_count, probe_count, 8, 2), np.float32)
    probe_knots = np.empty_like(probe_raw)
    probe_cost = np.empty((row_count, branch_count, round_count, probe_count), np.float32)
    probe_residual = np.empty(
        (row_count, branch_count, round_count, probe_count, residual_dim), np.float32
    )
    probe_clipped = np.empty_like(probe_raw, dtype=bool)
    fit_x = np.empty((row_count, branch_count, round_count, 33, 16), np.float32)
    fit_weight = np.empty((row_count, branch_count, round_count, 33), np.float32)
    cost_gradient = np.empty((row_count, branch_count, round_count, 16), np.float32)
    trajectory_response = np.empty(
        (row_count, branch_count, round_count, 16, residual_dim), np.float32
    )
    gn_step = np.empty_like(cost_gradient)
    response_directions = np.empty((row_count, branch_count, round_count, 3, 16), np.float32)
    cost_fit_error = np.empty((row_count, branch_count, round_count), np.float32)
    residual_fit_error = np.empty_like(cost_fit_error)
    direction_cosine = np.empty_like(cost_fit_error)
    gn_rho_uncapped = np.empty_like(cost_fit_error)
    proposal_delta_sigma = np.empty(
        (row_count, branch_count, round_count, proposal_count, 16), np.float32
    )
    proposal_raw = np.empty(
        (row_count, branch_count, round_count, proposal_count, 8, 2), np.float32
    )
    proposal_knots = np.empty_like(proposal_raw)
    proposal_cost = np.empty(
        (row_count, branch_count, round_count, proposal_count), np.float32
    )
    proposal_residual = np.empty(
        (row_count, branch_count, round_count, proposal_count, residual_dim), np.float32
    )
    proposal_clipped = np.empty_like(proposal_raw, dtype=bool)
    proposal_predicted_scalar_cost = np.empty_like(proposal_cost)
    proposal_predicted_trajectory_cost = np.empty_like(proposal_cost)
    selected_source = np.empty((row_count, branch_count, round_count), np.int8)
    selected_local_index = np.empty((row_count, branch_count, round_count), np.int16)
    probe_only_best_cost = np.empty_like(cost_fit_error)
    response_incremental_gain = np.empty_like(cost_fit_error)

    for row in range(row_count):
        current = data["current_action"][row]
        initial = np.stack(
            (
                np.broadcast_to(current, (8, 2)),
                np.linspace(current, np.zeros(2, np.float32), 8),
                np.zeros((8, 2), np.float32),
                data["prior_canonical_knots"][row],
                data["old_warm_knots"][row],
            )
        ).astype(np.float32)
        initial = np.clip(initial, low, high)
        initial_cost, initial_residual = evaluate(controller, data, row, initial, weights)
        centers[row, :, 0] = initial
        center_cost[row, :, 0] = initial_cost
        center_residual[row, :, 0] = initial_residual

        for round_index in range(round_count):
            basis = bases[round_index].reshape(16, 8, 2)
            radius = float(radii[round_index])
            maximum_step = float(maximum_steps[round_index])
            for branch in range(branch_count):
                incumbent = centers[row, branch, round_index]
                incumbent_cost = float(center_cost[row, branch, round_index])
                incumbent_residual = center_residual[row, branch, round_index]
                raw = np.stack(
                    [
                        incumbent + sign * radius * direction * sigma[None]
                        for direction in basis
                        for sign in (1.0, -1.0)
                    ]
                ).astype(np.float32)
                local = np.clip(raw, low, high)
                local_cost, local_residual = evaluate(controller, data, row, local, weights)
                probe_raw[row, branch, round_index] = raw
                probe_knots[row, branch, round_index] = local
                probe_cost[row, branch, round_index] = local_cost
                probe_residual[row, branch, round_index] = local_residual
                probe_clipped[row, branch, round_index] = np.abs(raw - local) > 1e-7

                fitted = fit_response(
                    incumbent,
                    incumbent_cost,
                    incumbent_residual,
                    local,
                    local_cost,
                    local_residual,
                    sigma,
                    ridge,
                    damping,
                    maximum_step,
                )
                fit_x[row, branch, round_index] = fitted["x"]
                fit_weight[row, branch, round_index] = fitted["weight"]
                cost_gradient[row, branch, round_index] = fitted["gradient"]
                trajectory_response[row, branch, round_index] = fitted["response"]
                gn_step[row, branch, round_index] = fitted["gn_step"]
                response_directions[row, branch, round_index] = np.stack(
                    (
                        fitted["cost_direction"],
                        fitted["trajectory_direction"],
                        fitted["blend_direction"],
                    )
                )
                cost_fit_error[row, branch, round_index] = fitted["cost_fit_error"]
                residual_fit_error[row, branch, round_index] = fitted["residual_fit_error"]
                direction_cosine[row, branch, round_index] = fitted[
                    "direction_cosine_before_alignment"
                ]
                gn_rho_uncapped[row, branch, round_index] = fitted["gn_step_rho_uncapped"]

                steps = []
                for direction_index in range(3):
                    base_step = (
                        fitted["gn_step"]
                        if direction_index == 1
                        else maximum_step
                        * response_directions[row, branch, round_index, direction_index]
                    )
                    for factor in line_factors:
                        steps.append(float(factor) * base_step)
                steps = np.stack(steps).astype(np.float32)
                raw_proposals = incumbent[None] + steps.reshape(-1, 8, 2) * sigma
                local_proposals = np.clip(raw_proposals, low, high).astype(np.float32)
                local_proposal_cost, local_proposal_residual = evaluate(
                    controller, data, row, local_proposals, weights
                )
                proposal_delta_sigma[row, branch, round_index] = steps
                proposal_raw[row, branch, round_index] = raw_proposals
                proposal_knots[row, branch, round_index] = local_proposals
                proposal_cost[row, branch, round_index] = local_proposal_cost
                proposal_residual[row, branch, round_index] = local_proposal_residual
                proposal_clipped[row, branch, round_index] = (
                    np.abs(raw_proposals - local_proposals) > 1e-7
                )
                proposal_predicted_scalar_cost[row, branch, round_index] = (
                    incumbent_cost + steps @ fitted["gradient"]
                )
                predicted_residual = (
                    incumbent_residual[None]
                    + steps.astype(np.float64) @ fitted["response"].astype(np.float64)
                )
                proposal_predicted_trajectory_cost[row, branch, round_index] = np.sum(
                    np.square(predicted_residual), axis=1
                )

                probe_best = min(incumbent_cost, float(np.min(local_cost)))
                proposal_best = float(np.min(local_proposal_cost))
                probe_only_best_cost[row, branch, round_index] = probe_best
                response_incremental_gain[row, branch, round_index] = max(
                    0.0, probe_best - proposal_best
                )
                combined_cost = np.concatenate(
                    ([incumbent_cost], local_cost, local_proposal_cost)
                )
                winner = int(np.argmin(combined_cost))
                if winner == 0:
                    next_center = incumbent
                    next_residual = incumbent_residual
                    source_code, local_index = 0, 0
                elif winner <= probe_count:
                    local_index = winner - 1
                    next_center = local[local_index]
                    next_residual = local_residual[local_index]
                    source_code = 1
                else:
                    local_index = winner - 1 - probe_count
                    next_center = local_proposals[local_index]
                    next_residual = local_proposal_residual[local_index]
                    source_code = 2
                centers[row, branch, round_index + 1] = next_center
                center_cost[row, branch, round_index + 1] = combined_cost[winner]
                center_residual[row, branch, round_index + 1] = next_residual
                selected_source[row, branch, round_index] = source_code
                selected_local_index[row, branch, round_index] = local_index
        print(
            f"remaining-state search {row + 1}/{row_count} "
            f"source_row={int(data['source_audit_row'][row])} "
            f"speed={int(data['speed_kph'][row])} episode={data['episode_id'][row]}",
            flush=True,
        )

    canonical_indices = np.flatnonzero(canonical_branch)
    canonical_local = np.argmin(center_cost[:, canonical_branch, -1], axis=1)
    canonical_global = canonical_indices[canonical_local]
    rows = np.arange(row_count)
    canonical_final_cost = center_cost[rows, canonical_global, -1]
    canonical_final_knots = centers[rows, canonical_global, -1]
    basin_counts = np.empty(row_count, np.int64)
    basin_separation = np.empty(row_count, np.float32)
    for row in range(row_count):
        basin_counts[row], basin_separation[row] = basin_count(
            centers[row, canonical_branch, -1],
            center_cost[row, canonical_branch, -1],
            sigma,
        )
    data.update(
        {
            "branch_name": branch_names,
            "branch_role": branch_roles,
            "round_basis": bases,
            "round_radius_sigma": radii,
            "round_maximum_step_sigma": maximum_steps,
            "line_factor": line_factors,
            "center_knots": centers,
            "center_cost": center_cost,
            "center_residual": center_residual,
            "probe_raw_knots": probe_raw,
            "probe_knots": probe_knots,
            "probe_cost": probe_cost,
            "probe_residual": probe_residual,
            "probe_clipped_mask": probe_clipped,
            "fit_x": fit_x,
            "fit_weight": fit_weight,
            "cost_gradient": cost_gradient,
            "trajectory_response": trajectory_response,
            "gauss_newton_step_sigma": gn_step,
            "response_direction": response_directions,
            "cost_relative_fit_error": cost_fit_error,
            "trajectory_relative_fit_error": residual_fit_error,
            "cost_trajectory_direction_cosine_before_alignment": direction_cosine,
            "gauss_newton_step_rho_uncapped": gn_rho_uncapped,
            "proposal_delta_sigma": proposal_delta_sigma,
            "proposal_raw_knots": proposal_raw,
            "proposal_knots": proposal_knots,
            "proposal_cost": proposal_cost,
            "proposal_residual": proposal_residual,
            "proposal_clipped_mask": proposal_clipped,
            "proposal_predicted_scalar_cost": proposal_predicted_scalar_cost,
            "proposal_predicted_trajectory_cost": proposal_predicted_trajectory_cost,
            "selected_source": selected_source,
            "selected_local_index": selected_local_index,
            "probe_only_best_cost": probe_only_best_cost,
            "response_incremental_gain": response_incremental_gain,
            "canonical_final_branch": canonical_global,
            "canonical_final_knots": canonical_final_knots,
            "canonical_final_cost": canonical_final_cost,
            "canonical_basin_count_within_10pct_separation_ge_0p5": basin_counts,
            "canonical_basin_min_separation_sigma_rms": basin_separation,
        }
    )
    return data


def combine(
    pilot: dict[str, np.ndarray], new: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if set(pilot) != set(new):
        missing = sorted(set(pilot) ^ set(new))
        raise AssertionError(f"pilot/new schema mismatch: {missing}")
    result = {}
    combined_rows = np.concatenate((pilot["source_audit_row"], new["source_audit_row"]))
    order = np.argsort(combined_rows)
    if not np.array_equal(combined_rows[order], np.arange(100)):
        raise AssertionError("combined source rows are not exactly 0..99")
    for name in pilot:
        if name in SHARED_FIELDS:
            if not np.array_equal(pilot[name], new[name]):
                raise AssertionError(f"shared field differs: {name}")
            result[name] = pilot[name]
        else:
            result[name] = np.concatenate((pilot[name], new[name]), axis=0)[order]
    for name in pilot:
        if name not in SHARED_FIELDS:
            if not np.array_equal(result[name][pilot["source_audit_row"]], pilot[name]):
                raise AssertionError(f"embedded pilot field changed: {name}")
    return result


def summarize(data: dict[str, np.ndarray], config: dict) -> tuple[dict, dict, str]:
    canonical = data["branch_role"] == "canonical"
    final = data["canonical_final_cost"]
    overall = grouped_metrics(data, np.ones(len(final), dtype=bool))
    first_two = data["response_incremental_gain"][:, canonical, :2] > 1e-6
    trajectory_improves = (
        np.min(data["proposal_cost"][..., 2:4], axis=-1)
        < data["probe_only_best_cost"] - 1e-6
    )
    first_two_fraction = float(np.mean(first_two))
    trajectory_fraction = float(np.mean(trajectory_improves[:, canonical]))
    aggregate_fraction = float(
        np.sum(data["prior_canonical_cost"] - final)
        / np.sum(data["prior_canonical_cost"])
    )
    speed_gains = {
        str(speed): float(
            np.mean(
                data["prior_canonical_cost"][data["speed_kph"] == speed]
                - final[data["speed_kph"] == speed]
            )
        )
        for speed in sorted(np.unique(data["speed_kph"]))
    }
    gate_config = config["pre_registered_expand_to_100_state_gates"]
    gates = {
        "canonical_final_mean_cost_no_greater_than_old_warm": bool(
            np.mean(final) <= np.mean(data["old_warm_cost"]) + 1e-8
        ),
        "canonical_final_warm_regression_fraction_le_0p25": bool(
            overall["warm_regression_fraction"]
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
    response_pass = gates[
        "first_two_round_response_proposal_improvement_fraction_ge_0p50"
    ] and gates["trajectory_response_realized_improvement_positive_fraction_ge_0p50"]
    if all(gates.values()):
        decision = "FULL100_LANDSCAPE_READY_FOR_ABSOLUTE_CRITIC_REPLAY_PENDING_VALIDATION"
    elif response_pass:
        decision = "FULL100_RESPONSE_WORKS_COST_GATE_FAIL_NO_CRITIC_REPLAY"
    else:
        decision = "FULL100_RESPONSE_FAIL_STOP"
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "performance_decision": decision,
        "row_count": 100,
        "episode_count": int(len(np.unique(data["episode_id"]))),
        "embedded_pilot_state_count": 20,
        "new_state_count": 80,
        "rollouts_per_state": 765,
        "new_query_rollouts": 61200,
        "represented_query_rollouts": 76500,
        "overall": overall,
        "center_cost_by_round": {
            str(round_index): stats(
                np.min(data["center_cost"][:, canonical, round_index], axis=1)
            )
            for round_index in range(5)
        },
        "branch_final_cost": {
            str(data["branch_name"][branch]): stats(data["center_cost"][:, branch, -1])
            for branch in range(5)
        },
        "response": {
            "first_two_round_proposal_improvement_fraction": first_two_fraction,
            "trajectory_proposal_improvement_fraction": trajectory_fraction,
            "incremental_gain": stats(data["response_incremental_gain"][:, canonical]),
            "cost_relative_fit_error": stats(
                data["cost_relative_fit_error"][:, canonical]
            ),
            "trajectory_relative_fit_error": stats(
                data["trajectory_relative_fit_error"][:, canonical]
            ),
        },
        "aggregate_gain_vs_prior_canonical_fraction": aggregate_fraction,
        "speed_mean_gain_vs_prior_canonical": speed_gains,
        "canonical_basin_count": stats(
            data["canonical_basin_count_within_10pct_separation_ge_0p5"]
        ),
        "pre_registered_performance_gates": gates,
        "by_speed_kph": {
            str(speed): grouped_metrics(data, data["speed_kph"] == speed)
            for speed in sorted(np.unique(data["speed_kph"]))
        },
        "by_variant_index": {
            str(variant): grouped_metrics(data, data["variant_index"] == variant)
            for variant in sorted(np.unique(data["variant_index"]))
        },
        "by_fold": {
            str(fold): grouped_metrics(data, data["fold_id"] == fold)
            for fold in range(5)
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }
    return summary, gates, decision


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    expand_config = json.loads(config_path.read_text())
    source = Path(expand_config["source_audit"]).resolve()
    pilot_path = Path(expand_config["pilot_artifact"]).resolve()
    output = Path(expand_config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if expand_config["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    if expand_config.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("DBM fields or labels are forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    pilot_manifest_path = pilot_path / "manifest.json"
    pilot_validation_path = pilot_path / "validation.json"
    pilot_landscape_path = pilot_path / "landscape.npz"
    pilot_manifest = json.loads(pilot_manifest_path.read_text())
    pilot_validation = json.loads(pilot_validation_path.read_text())
    if pilot_validation["qualification"] != "QUERY_FORWARD_RESPONSE_LANDSCAPE_INDEPENDENT_PASS":
        raise AssertionError("pilot did not independently pass")
    if sha256(pilot_landscape_path) != pilot_manifest["landscape_sha256"]:
        raise AssertionError("pilot landscape hash mismatch")
    with np.load(pilot_landscape_path, allow_pickle=False) as archive:
        pilot = {name: np.asarray(archive[name]) for name in archive.files}
    if len(pilot["source_audit_row"]) != int(expand_config["pilot_state_count"]):
        raise AssertionError("pilot state count mismatch")
    all_rows = np.arange(int(expand_config["full_state_count"]), dtype=np.int64)
    remaining = np.setdiff1d(all_rows, pilot["source_audit_row"], assume_unique=True)
    if len(remaining) != int(expand_config["new_state_count"]):
        raise AssertionError("remaining state count mismatch")
    new = load_source_rows(source, remaining)
    if len(np.unique(new["episode_id"])) != len(remaining):
        raise AssertionError("remaining states are not episode independent")
    if [int(np.sum(new["fold_id"] == fold)) for fold in range(5)] != [16] * 5:
        raise AssertionError("remaining fold balance failed")

    pilot_config_path = pilot_path / "config.json"
    search_config = json.loads(pilot_config_path.read_text())
    source_manifest_path = source / "manifest.json"
    source_validation_path = source / "validation.json"
    source_audit_path = source / "audit.npz"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_validation = json.loads(source_validation_path.read_text())
    if source_validation["qualification"] != "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS":
        raise AssertionError("source audit did not independently pass")
    if sha256(source_audit_path) != source_manifest["audit_sha256"]:
        raise AssertionError("source audit hash mismatch")
    collection, parent_t0 = find_collection(source_manifest)
    collection_manifest_path = collection / "manifest.json"
    collection_manifest = json.loads(collection_manifest_path.read_text())
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    checkpoint = Path(source_manifest["query_checkpoint"])
    model = QueryDeploymentModel.from_checkpoint(checkpoint, device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(model), params, device=str(device)
    )
    new = run_search(new, controller, search_config)
    full = combine(pilot, new)
    if [int(np.sum(full["fold_id"] == fold)) for fold in range(5)] != [20] * 5:
        raise AssertionError("full fold balance failed")
    summary, gates, decision = summarize(full, search_config)

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    shutil.copy2(pilot_config_path, output / "search_contract.json")
    landscape_path = output / "landscape.npz"
    np.savez_compressed(landscape_path, **full)
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    rows_path = output / "rows.csv"
    with rows_path.open("w", newline="") as stream:
        fields = [
            "source_audit_row",
            "episode_id",
            "speed_kph",
            "variant_index",
            "fold_id",
            "prior_canonical_cost",
            "old_warm_cost",
            "old_t0_cost",
            "old_fullrank_cost",
            "canonical_final_branch",
            "canonical_final_cost",
            "canonical_basin_count_within_10pct_separation_ge_0p5",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in range(100):
            writer.writerow({name: np.asarray(full[name][row]).item() for name in fields})
    manifest = {
        "schema_version": "query-forward-response-landscape-full100-v1",
        "dataset_type": "train-only-pure-query-forward-response-landscape-full100",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "performance_decision": decision,
        "config_sha256": sha256(config_path),
        "search_contract_sha256": sha256(pilot_config_path),
        "source_audit": str(source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_validation_sha256": sha256(source_validation_path),
        "source_audit_sha256": sha256(source_audit_path),
        "embedded_pilot": str(pilot_path),
        "embedded_pilot_manifest_sha256": sha256(pilot_manifest_path),
        "embedded_pilot_validation_sha256": sha256(pilot_validation_path),
        "embedded_pilot_landscape_sha256": sha256(pilot_landscape_path),
        "source_collection": str(collection),
        "source_collection_manifest_sha256": sha256(collection_manifest_path),
        "parent_t0": str(parent_t0),
        "query_checkpoint": str(checkpoint),
        "query_checkpoint_sha256": sha256(checkpoint),
        "landscape_sha256": sha256(landscape_path),
        "summary_sha256": sha256(summary_path),
        "rows_sha256": sha256(rows_path),
        "row_count": 100,
        "embedded_pilot_state_count": 20,
        "new_state_count": 80,
        "rollouts_per_state": 765,
        "new_query_rollouts": 61200,
        "represented_query_rollouts": 76500,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "response_attestation": (
            "New rows use only same-state frozen-Query forward probes and real-Query "
            "proposal scoring; embedded pilot rows are unchanged."
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "limitations": [
            "This is a 100-state train-only landscape, not a teacher or Actor dataset.",
            "Only 80 states received new rollouts; 20 independently validated pilot states are embedded unchanged.",
            "Query J50 is a model-space objective and not physical validation.",
            "Formal validation/test, wrapper, and closed loop remain sealed.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "gates": gates, "overall": summary["overall"], "response": summary["response"]}, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
