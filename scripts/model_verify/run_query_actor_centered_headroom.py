#!/usr/bin/env python3
"""Audit residual Query-cost headroom around the three selected OAC Actors."""

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
from run_query_forward_response_landscape_pilot import (  # noqa: E402
    basis_bank,
    evaluate,
    fit_response,
    sha256,
    stats,
)
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    load_inputs,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_centered_headroom_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def subset(data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in data.items():
        array = np.asarray(value)
        if array.ndim and len(array) == len(data["state"]):
            result[name] = array[rows]
    return result


def headroom_metrics(
    actor_cost: np.ndarray,
    best_cost: np.ndarray,
    actor_knots: np.ndarray,
    best_knots: np.ndarray,
    sigma: np.ndarray,
    speed_kph: np.ndarray,
    variant_index: np.ndarray,
) -> dict[str, Any]:
    actor_cost = np.asarray(actor_cost, np.float64)
    best_cost = np.asarray(best_cost, np.float64)
    gain = actor_cost - best_cost
    relative = gain / np.maximum(actor_cost, 1e-12)
    movement = np.sqrt(np.mean(np.square((best_knots - actor_knots) / sigma), axis=(1, 2)))

    def group(mask: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(mask.sum()),
            "actor_cost": stats(actor_cost[mask]),
            "best_cost": stats(best_cost[mask]),
            "gain": stats(gain[mask]),
            "relative_reduction": stats(relative[mask]),
            "aggregate_residual_reduction": float(gain[mask].sum() / actor_cost[mask].sum()),
            "improved_fraction": float(np.mean(gain[mask] > 1e-5)),
            "movement_sigma_rms": stats(movement[mask]),
        }

    report = group(np.ones(len(actor_cost), dtype=bool))
    report["by_speed_kph"] = {
        str(int(speed)): group(speed_kph == speed) for speed in np.unique(speed_kph)
    }
    report["by_speed_variant"] = {
        f"{int(speed)}:{int(variant)}": group(
            (speed_kph == speed) & (variant_index == variant)
        )
        for speed in np.unique(speed_kph)
        for variant in np.unique(variant_index[speed_kph == speed])
    }
    return report


def run_seed(
    seed: int,
    config: dict,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    controller: TorchMPPIController,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    source = Path(config["sources"]["actor_oac"])
    checkpoint_path = source / f"seed_{seed}" / "checkpoint.pt"
    arrays_path = source / f"seed_{seed}" / "oac_arrays.npz"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    actor_inputs = load_inputs(data, payload["actor_normalization"])
    actor_knots = actor_predict(actor, actor_inputs, rows, device)

    with np.load(arrays_path, allow_pickle=False) as archive:
        source_rows = np.asarray(archive["selection_indices"], np.int64)
        source_action = np.asarray(
            archive["selection_round_action"][int(payload["selected_round"])], np.float32
        )
        source_cost = np.asarray(
            archive["selection_round_cost"][int(payload["selected_round"])], np.float32
        )
    if not np.array_equal(source_rows, rows):
        raise AssertionError("source selection rows differ from frozen inner split")
    actor_action_error = float(np.max(np.abs(actor_knots - source_action)))
    if actor_action_error != 0.0:
        raise AssertionError(f"selected Actor action reload error {actor_action_error}")

    local = subset(data, rows)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    bases = basis_bank()
    radii = np.asarray(config["probe_contract"]["radius_sigma_by_round"], np.float32)
    maximum_steps = np.asarray(
        config["response_contract"]["maximum_step_sigma_by_round"], np.float32
    )
    line_factors = np.asarray(config["response_contract"]["line_factors"], np.float32)
    ridge = float(config["response_contract"]["fit_ridge"])
    damping = float(config["response_contract"]["gauss_newton_damping"])
    rounds = int(config["round_count"])
    row_count = len(rows)
    probe_count, proposal_count, residual_dim = 32, 6, 300

    centers = np.empty((row_count, rounds + 1, 8, 2), np.float32)
    center_cost = np.empty((row_count, rounds + 1), np.float32)
    center_residual = np.empty((row_count, rounds + 1, residual_dim), np.float32)
    probe_raw = np.empty((row_count, rounds, probe_count, 8, 2), np.float32)
    probe_knots = np.empty_like(probe_raw)
    probe_cost = np.empty((row_count, rounds, probe_count), np.float32)
    probe_residual = np.empty((row_count, rounds, probe_count, residual_dim), np.float32)
    probe_clipped = np.empty_like(probe_raw, dtype=bool)
    cost_gradient = np.empty((row_count, rounds, 16), np.float32)
    trajectory_response = np.empty((row_count, rounds, 16, residual_dim), np.float32)
    gn_step = np.empty_like(cost_gradient)
    response_directions = np.empty((row_count, rounds, 3, 16), np.float32)
    cost_fit_error = np.empty((row_count, rounds), np.float32)
    residual_fit_error = np.empty_like(cost_fit_error)
    proposal_delta_sigma = np.empty((row_count, rounds, proposal_count, 16), np.float32)
    proposal_raw = np.empty((row_count, rounds, proposal_count, 8, 2), np.float32)
    proposal_knots = np.empty_like(proposal_raw)
    proposal_cost = np.empty((row_count, rounds, proposal_count), np.float32)
    proposal_residual = np.empty((row_count, rounds, proposal_count, residual_dim), np.float32)
    proposal_clipped = np.empty_like(proposal_raw, dtype=bool)
    selected_source = np.empty((row_count, rounds), np.int8)
    selected_local_index = np.empty((row_count, rounds), np.int16)

    for row in range(row_count):
        initial_cost, initial_residual = evaluate(
            controller, local, row, actor_knots[row : row + 1], weights
        )
        centers[row, 0] = actor_knots[row]
        center_cost[row, 0] = initial_cost[0]
        center_residual[row, 0] = initial_residual[0]
        for round_index in range(rounds):
            incumbent = centers[row, round_index]
            incumbent_cost = float(center_cost[row, round_index])
            incumbent_residual = center_residual[row, round_index]
            basis = bases[round_index].reshape(16, 8, 2)
            raw = np.stack([
                incumbent + sign * float(radii[round_index]) * direction * sigma[None]
                for direction in basis for sign in (1.0, -1.0)
            ]).astype(np.float32)
            probes = np.clip(raw, low, high).astype(np.float32)
            costs, residuals = evaluate(controller, local, row, probes, weights)
            fitted = fit_response(
                incumbent, incumbent_cost, incumbent_residual,
                probes, costs, residuals, sigma, ridge, damping,
                float(maximum_steps[round_index]),
            )
            directions = np.stack((
                fitted["cost_direction"],
                fitted["trajectory_direction"],
                fitted["blend_direction"],
            ))
            steps = []
            for direction_index in range(3):
                base_step = (
                    fitted["gn_step"] if direction_index == 1
                    else float(maximum_steps[round_index]) * directions[direction_index]
                )
                for factor in line_factors:
                    steps.append(float(factor) * base_step)
            steps = np.stack(steps).astype(np.float32)
            raw_proposals = incumbent[None] + steps.reshape(-1, 8, 2) * sigma[None, None]
            proposals = np.clip(raw_proposals, low, high).astype(np.float32)
            proposal_costs, proposal_residuals = evaluate(
                controller, local, row, proposals, weights
            )
            combined_cost = np.concatenate(([incumbent_cost], costs, proposal_costs))
            winner = int(np.argmin(combined_cost))
            if winner == 0:
                next_knots, next_residual, source_code, local_index = (
                    incumbent, incumbent_residual, 0, 0
                )
            elif winner <= probe_count:
                local_index = winner - 1
                next_knots, next_residual, source_code = (
                    probes[local_index], residuals[local_index], 1
                )
            else:
                local_index = winner - 1 - probe_count
                next_knots, next_residual, source_code = (
                    proposals[local_index], proposal_residuals[local_index], 2
                )
            centers[row, round_index + 1] = next_knots
            center_cost[row, round_index + 1] = combined_cost[winner]
            center_residual[row, round_index + 1] = next_residual
            probe_raw[row, round_index] = raw
            probe_knots[row, round_index] = probes
            probe_cost[row, round_index] = costs
            probe_residual[row, round_index] = residuals
            probe_clipped[row, round_index] = np.abs(raw - probes) > 1e-7
            cost_gradient[row, round_index] = fitted["gradient"]
            trajectory_response[row, round_index] = fitted["response"]
            gn_step[row, round_index] = fitted["gn_step"]
            response_directions[row, round_index] = directions
            cost_fit_error[row, round_index] = fitted["cost_fit_error"]
            residual_fit_error[row, round_index] = fitted["residual_fit_error"]
            proposal_delta_sigma[row, round_index] = steps
            proposal_raw[row, round_index] = raw_proposals
            proposal_knots[row, round_index] = proposals
            proposal_cost[row, round_index] = proposal_costs
            proposal_residual[row, round_index] = proposal_residuals
            proposal_clipped[row, round_index] = np.abs(raw_proposals - proposals) > 1e-7
            selected_source[row, round_index] = source_code
            selected_local_index[row, round_index] = local_index
        if row == 0 or (row + 1) % 20 == 0:
            print(
                f"seed={seed} actor-headroom {row + 1}/{row_count} "
                f"episode={local['episode_id'][row]} speed={int(local['speed_kph'][row])}",
                flush=True,
            )

    actor_cost_error = float(np.max(np.abs(center_cost[:, 0] - source_cost)))
    if actor_cost_error > 1e-6:
        raise AssertionError(f"selected Actor Query cost reload error {actor_cost_error}")
    round_metrics = [
        headroom_metrics(
            center_cost[:, 0], center_cost[:, round_index],
            centers[:, 0], centers[:, round_index], sigma,
            local["speed_kph"], local["variant_index"],
        )
        for round_index in range(rounds + 1)
    ]
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir()
    result_arrays = seed_dir / "headroom_arrays.npz"
    np.savez_compressed(
        result_arrays,
        selection_indices=rows,
        episode_id=local["episode_id"],
        speed_kph=local["speed_kph"],
        variant_index=local["variant_index"],
        actor_knots=actor_knots,
        actor_source_cost=source_cost,
        centers=centers,
        center_cost=center_cost,
        center_residual=center_residual,
        probe_raw=probe_raw,
        probe_knots=probe_knots,
        probe_cost=probe_cost,
        probe_residual=probe_residual,
        probe_clipped=probe_clipped,
        cost_gradient=cost_gradient,
        trajectory_response=trajectory_response,
        gn_step=gn_step,
        response_directions=response_directions,
        cost_fit_error=cost_fit_error,
        residual_fit_error=residual_fit_error,
        proposal_delta_sigma=proposal_delta_sigma,
        proposal_raw=proposal_raw,
        proposal_knots=proposal_knots,
        proposal_cost=proposal_cost,
        proposal_residual=proposal_residual,
        proposal_clipped=proposal_clipped,
        selected_source=selected_source,
        selected_local_index=selected_local_index,
    )
    return {
        "seed": seed,
        "source_selected_round": int(payload["selected_round"]),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_arrays": str(arrays_path.resolve()),
        "source_arrays_sha256": sha256(arrays_path),
        "actor_action_reload_max_abs_error": actor_action_error,
        "actor_cost_reload_max_abs_error": actor_cost_error,
        "round_metrics": round_metrics,
        "arrays": str(result_arrays.resolve()),
        "arrays_sha256": sha256(result_arrays),
    }


def pooled_metrics(records: list[dict[str, Any]], config: dict) -> list[dict[str, Any]]:
    arrays = [np.load(record["arrays"], allow_pickle=False) for record in records]
    try:
        sigma = np.asarray(config["noise_sigma"], np.float32)
        return [
            headroom_metrics(
                np.concatenate([value["center_cost"][:, 0] for value in arrays]),
                np.concatenate([value["center_cost"][:, round_index] for value in arrays]),
                np.concatenate([value["centers"][:, 0] for value in arrays]),
                np.concatenate([value["centers"][:, round_index] for value in arrays]),
                sigma,
                np.concatenate([value["speed_kph"] for value in arrays]),
                np.concatenate([value["variant_index"] for value in arrays]),
            )
            for round_index in range(int(config["round_count"]) + 1)
        ]
    finally:
        for value in arrays:
            value.close()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if (config["formal_validation_or_test_consumed"]
            or config["dbm_fields_or_labels_consumed"]
            or config["analytic_dbm_or_query_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source = Path(config["sources"]["actor_oac"])
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text())
    if validation["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source Actor OAC did not independently pass")
    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    rows = np.flatnonzero(data["fold_id"] == int(config["split_contract"]["inner_selection_fold"]))
    if len(rows) != int(config["split_contract"]["expected_state_count"]):
        raise AssertionError("inner state-count contract failed")
    if np.any(data["fold_id"][rows] == int(config["split_contract"]["outer_fold"])):
        raise AssertionError("outer fold leaked into audit")

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    output.mkdir(parents=True)
    records = [
        run_seed(int(seed), config, data, rows, controller, output, device)
        for seed in config["actor_seeds"]
    ]
    pooled = pooled_metrics(records, config)
    final = pooled[-1]
    aggregate = float(final["aggregate_residual_reduction"])
    majority = float(final["improved_fraction"]) > 0.5
    if aggregate < 0.02:
        decision = "STOP_BROAD_QUERY_ACTOR_EXPANSION_LOW_HEADROOM"
    elif aggregate < 0.05 or not majority:
        decision = "QUERY_ACTOR_HAS_SMALL_HEADROOM_ALLOW_SINGLE_VARIABLE_AB"
    else:
        decision = "QUERY_ACTOR_HAS_MATERIAL_HEADROOM_ADVANCE_SINGLE_VARIABLE_AB"
    summary = {
        "qualification": "QUERY_ACTOR_CENTERED_HEADROOM_AUDIT_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled_round_metrics": pooled,
        "decision": decision,
        "decision_inputs": {
            "final_aggregate_residual_reduction": aggregate,
            "final_improved_fraction": float(final["improved_fraction"]),
            "majority_improved": majority,
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "actor_or_critic_trained": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-actor-centered-headroom-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_actor_oac": str(source.resolve()),
        "source_actor_validation_sha256": sha256(validation_path),
        "source_replay": config["sources"]["absolute_replay"],
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(output / "summary.json"),
        "result_arrays_sha256": {f"seed_{r['seed']}": r["arrays_sha256"] for r in records},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "actor_mean_cost": pooled[0]["actor_cost"]["mean"],
        "best_mean_cost": final["best_cost"]["mean"],
        "aggregate_residual_reduction": aggregate,
        "paired_median_relative_reduction": final["relative_reduction"]["median"],
        "improved_fraction": final["improved_fraction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
