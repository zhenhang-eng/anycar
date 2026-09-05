#!/usr/bin/env python3
"""Diagnose Query OAC hard slices using the fixed-1e-5 selected checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_query_oac_gamma1_k_scan as base
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_actor_aggregation_microstep import actor_gradient, objective_weights, predict_conservative_log, vector_cosine
from run_query_single_center_oac20to1 import (
    actor_from_payload,
    actor_predict,
    candidate_ranking_metrics,
    direct_cost,
    distribution,
    load_inputs,
    physical_critic_value,
    response_bank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_fixed_lr1e5_hard_slice_diagnostic_config_20260903_v1.json"


def deterministic_contract() -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "mode": "warn_only",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def slice_mask(data: dict[str, np.ndarray], rows: np.ndarray, name: str) -> np.ndarray:
    speed, variant = (int(value) for value in name.split(":"))
    return (data["speed_kph"][rows] == speed) & (data["variant_index"][rows] == variant)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def critic_values(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    row: int,
    action: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rows = np.full(len(action), row, np.int64)
    action_tensor = torch.from_numpy(action).to(device)
    values = []
    with torch.no_grad():
        for critic, training in zip(critics, trainings):
            values.append(physical_critic_value(critic, training, inputs, rows, action_tensor, device).cpu().numpy())
    return np.maximum(values[0], values[1]).astype(np.float32)


def critic_action_gradient(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    row: int,
    center: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rows = np.asarray([row], np.int64)
    action = torch.from_numpy(center[None]).to(device).requires_grad_(True)
    values = [physical_critic_value(critic, training, inputs, rows, action, device) for critic, training in zip(critics, trainings)]
    conservative = torch.maximum(values[0], values[1])
    gradient = torch.autograd.grad(conservative.sum(), action)[0]
    return (gradient * torch.from_numpy(sigma).to(device)).detach().cpu().numpy().reshape(16).astype(np.float32)


def fresh_bank_record(
    *,
    controller: base.TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    center: np.ndarray,
    center_name: str,
    radius: float,
    basis: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, float],
    config: dict,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions, costs, raw, clipped = response_bank(
        controller, data, row, center, radius, basis, sigma, weights, config,
    )
    predicted = critic_values(critics, trainings, inputs, row, actions, device)
    x = ((actions[1:33] - center[None]) / sigma).reshape(32, 16).astype(np.float64)
    y = np.log1p(costs[1:33].astype(np.float64)) - np.log1p(float(costs[0]))
    ridge = float(config["fresh_local_response"]["finite_difference_ridge"])
    true_gradient = np.linalg.solve(x.T @ x + ridge * np.eye(16), x.T @ y).astype(np.float32)
    predicted_gradient = critic_action_gradient(critics, trainings, inputs, row, center, sigma, device)
    true_norm = float(np.linalg.norm(true_gradient))
    predicted_norm = float(np.linalg.norm(predicted_gradient))
    gradient_cosine = vector_cosine(true_gradient, predicted_gradient)
    true_delta = np.log1p(costs[1:].astype(np.float64)) - np.log1p(float(costs[0]))
    predicted_delta = predicted[1:].astype(np.float64) - float(predicted[0])
    material = np.abs(true_delta) > 1e-7
    sign_accuracy = float(np.mean(np.sign(true_delta[material]) == np.sign(predicted_delta[material]))) if np.any(material) else 1.0
    best = int(np.argmin(costs))
    critic_best = int(np.argmin(predicted))
    denominator = max(float(costs[0] - costs[best]), 1e-12)
    recovery = float((costs[0] - costs[critic_best]) / denominator) if costs[0] > costs[best] + 1e-8 else 0.0
    report = {
        "row": row,
        "episode_id": str(data["episode_id"][row]),
        "road_name": str(data["road_name"][row]),
        "control_step": int(data["control_step"][row]),
        "center": center_name,
        "radius_sigma": radius,
        "center_cost": float(costs[0]),
        "warm_cost": float(data["warm_cost"][row]),
        "best_cost": float(costs[best]),
        "best_gain_vs_center": float(costs[0] - costs[best]),
        "best_gain_vs_warm": float(data["warm_cost"][row] - costs[best]),
        "best_index": best,
        "critic_selected_index": critic_best,
        "critic_selected_cost": float(costs[critic_best]),
        "critic_bank_gain_recovery": recovery,
        "critic_log_cost_pearson": correlation(predicted, np.log1p(costs)),
        "critic_center_relative_sign_accuracy": sign_accuracy,
        "fd_gradient_cosine": gradient_cosine,
        "fd_gradient_norm_ratio": predicted_norm / max(true_norm, 1e-12),
        "fd_true_gradient_norm": true_norm,
        "fd_predicted_gradient_norm": predicted_norm,
        "clipped_fraction": float(np.mean(clipped)),
    }
    arrays = {
        "action": actions,
        "raw_action": raw,
        "cost": costs,
        "clipped": clipped,
        "critic_log_cost": predicted,
        "fd_true_gradient_z": true_gradient,
        "fd_predicted_gradient_z": predicted_gradient,
    }
    return report, arrays


def warm_report(cost: np.ndarray, data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, Any]:
    return base.warm_relative_metrics(cost, data["warm_cost"][rows], data["speed_kph"][rows], data["variant_index"][rows])


def offline_coverage(data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, Any]:
    best, count = [], []
    for row in rows:
        valid = data["candidate_valid_mask"][row]
        count.append(int(valid.sum()))
        best.append(float(np.min(data["candidate_cost"][row, valid])))
    best_array = np.asarray(best, np.float32)
    return {
        "state_count": len(rows),
        "candidate_count": distribution(np.asarray(count, np.float64)),
        "best_candidate": warm_report(best_array, data, rows),
    }


def actor_encoder_features(
    actor: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return the deployable no-anchor encoder feature for every replay row."""
    output = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(inputs[0]), batch_size):
            tensors = tuple(
                torch.from_numpy(value[start : start + batch_size]).to(device)
                for value in inputs
            )
            output.append(
                actor.encoder(
                    tensors[0], tensors[1], tensors[2],
                    torch.zeros_like(tensors[3]),
                    torch.zeros_like(tensors[4]),
                    torch.zeros_like(tensors[5]),
                ).cpu().numpy()
            )
    return np.concatenate(output).astype(np.float32)


def nearest_rms(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_flat = np.asarray(query, np.float32).reshape(len(query), -1)
    reference_flat = np.asarray(reference, np.float32).reshape(len(reference), -1)
    distance = np.sqrt(np.mean(np.square(query_flat[:, None] - reference_flat[None]), axis=2))
    nearest = np.argmin(distance, axis=1)
    return distance[np.arange(len(query)), nearest].astype(np.float32), nearest.astype(np.int64)


def input_neighborhood_report(
    data: dict[str, np.ndarray],
    inputs: tuple[np.ndarray, ...],
    feature: np.ndarray,
    fit: np.ndarray,
    selection: np.ndarray,
    name: str,
) -> dict[str, Any]:
    fit_rows = fit[slice_mask(data, fit, name)]
    selection_rows = selection[slice_mask(data, selection, name)]
    component_names = ("history", "reference", "current")
    components: dict[str, Any] = {}
    for component_name, value in zip(component_names, inputs[:3]):
        distance, nearest = nearest_rms(value[selection_rows], value[fit_rows])
        components[component_name] = {
            "nearest_fit_rms": distribution(distance),
            "per_row": [
                {
                    "row": int(row),
                    "control_step": int(data["control_step"][row]),
                    "nearest_fit_row": int(fit_rows[nearest[position]]),
                    "distance_rms": float(distance[position]),
                }
                for position, row in enumerate(selection_rows)
            ],
        }
    fit_mean = feature[fit].mean(axis=0, keepdims=True)
    fit_std = feature[fit].std(axis=0, keepdims=True) + 1e-6
    standardized = (feature - fit_mean) / fit_std
    distance, nearest = nearest_rms(standardized[selection_rows], standardized[fit_rows])
    return {
        "fit_rows": fit_rows.tolist(),
        "inner_rows": selection_rows.tolist(),
        "normalized_actor_inputs": components,
        "selected_actor_encoder_feature_global_fit_standardized": {
            "nearest_fit_rms": distribution(distance),
            "per_row": [
                {
                    "row": int(row),
                    "control_step": int(data["control_step"][row]),
                    "nearest_fit_row": int(fit_rows[nearest[position]]),
                    "nearest_fit_episode_id": str(data["episode_id"][fit_rows[nearest[position]]]),
                    "nearest_fit_control_step": int(data["control_step"][fit_rows[nearest[position]]]),
                    "distance_rms": float(distance[position]),
                }
                for position, row in enumerate(selection_rows)
            ],
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed boundary violation")
    deterministic = deterministic_contract()
    source_dir = Path(config["sources"]["lr_decay_160round"]).resolve()
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    source_summary = json.loads((source_dir / "summary.json").read_text())
    source_validation = json.loads((source_dir / "validation.json").read_text())
    if source_validation["qualification"] != "QUERY_OAC_GAMMA1_K16_LR_DECAY_160ROUND_INDEPENDENT_PASS":
        raise AssertionError("source LR schedule artifact is not independently qualified")
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = base.QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = base.TorchMPPIController(
        base.TorchQueryRolloutBackend(query),
        base.TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["noise_sigma"], np.float32).reshape(1, 2)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    oof = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    slices = [config["target_slice"], *config["control_slices"]]
    all_strata = [
        f"{int(speed)}:{int(variant)}"
        for speed in sorted(np.unique(data["speed_kph"][fit]).tolist())
        for variant in sorted(np.unique(data["variant_index"][fit]).tolist())
    ]
    fold_counts = {
        str(fold): {name: int(np.sum(slice_mask(data, np.flatnonzero(data["fold_id"] == fold), name))) for name in all_strata}
        for fold in range(5)
    }
    coverage = {
        name: offline_coverage(data, fit[slice_mask(data, fit, name)])
        for name in all_strata
    }
    output.mkdir(parents=True)
    records = []
    bases = base.basis_bank()
    fresh_arrays_by_seed: dict[int, dict[str, list[np.ndarray]]] = {}
    selected_inner_action_by_seed: dict[int, np.ndarray] = {}
    selected_inner_cost_by_seed: dict[int, np.ndarray] = {}
    for seed in config["seeds"]:
        seed = int(seed)
        source_record = next(item for item in source_summary["records"] if item["arm"] == config["source_arm"] and int(item["seed"]) == seed)
        checkpoint_path = Path(source_record["checkpoint"])
        source_arrays_path = Path(source_record["arrays"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(source_arrays_path, allow_pickle=False) as archive:
            source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload({"actor_training": checkpoint["actor_training"], "selected_actor_state_dict": checkpoint["selected_actor_state_dict"]}, "selected_actor_state_dict", device)
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        fit_action = actor_predict(actor, inputs, fit, device)
        fit_cost = direct_cost(controller, data, fit, fit_action, weights)
        selection_action = actor_predict(actor, inputs, selection, device)
        selection_cost = direct_cost(controller, data, selection, selection_action, weights)
        selected_inner_action_by_seed[seed] = selection_action
        selected_inner_cost_by_seed[seed] = selection_cost
        encoder_feature = actor_encoder_features(
            actor, inputs, device, int(config["actor_gradient"]["batch_size"])
        )
        online_selected = source_arrays["round"] <= int(checkpoint["selected_round"])
        online_reports = {}
        visit_count_by_stratum = {}
        for name in all_strata:
            local_rows = fit[slice_mask(data, fit, name)]
            mask = online_selected & np.isin(source_arrays["state_index"], local_rows)
            visit_count_by_stratum[name] = int(len(np.unique(source_arrays["group"][mask])))
            online_reports[name] = candidate_ranking_metrics(
                critics, trainings, inputs,
                source_arrays["state_index"][mask], source_arrays["action"][mask],
                source_arrays["cost"][mask], source_arrays["group"][mask], device,
            )
        conservative_log = predict_conservative_log(actor, critics, trainings, inputs, fit, device, int(config["actor_gradient"]["batch_size"]))
        global_weight = objective_weights(conservative_log, float(config["actor_gradient"]["gamma"]), float(config["actor_gradient"]["weight_maximum"]))
        global_gradient, parameter_names, parameter_offsets = actor_gradient(actor, critics, trainings, inputs, fit, global_weight, device, int(config["actor_gradient"]["batch_size"]))
        stratum_gradients = {}
        for name in all_strata:
            mask = slice_mask(data, fit, name)
            gradient, names, offsets = actor_gradient(actor, critics, trainings, inputs, fit[mask], global_weight[mask], device, int(config["actor_gradient"]["batch_size"]))
            if names != parameter_names or not np.array_equal(offsets, parameter_offsets):
                raise AssertionError("Actor parameter layout changed")
            stratum_gradients[name] = gradient
        target_mask = slice_mask(data, fit, config["target_slice"])
        complement_gradient, _, _ = actor_gradient(actor, critics, trainings, inputs, fit[~target_mask], global_weight[~target_mask], device, int(config["actor_gradient"]["batch_size"]))
        gradient_report = {
            "global_norm": float(np.linalg.norm(global_gradient)),
            "target_norm": float(np.linalg.norm(stratum_gradients[config["target_slice"]])),
            "complement_norm": float(np.linalg.norm(complement_gradient)),
            "target_vs_global_cosine": vector_cosine(stratum_gradients[config["target_slice"]], global_gradient),
            "target_vs_complement_cosine": vector_cosine(stratum_gradients[config["target_slice"]], complement_gradient),
            "target_vs_stratum_cosine": {name: vector_cosine(stratum_gradients[config["target_slice"]], stratum_gradients[name]) for name in all_strata},
        }
        input_neighborhood = {
            name: input_neighborhood_report(
                data, inputs, encoder_feature, fit, selection, name
            )
            for name in all_strata
        }
        evaluations = {}
        for name in slices:
            fit_rows = fit[slice_mask(data, fit, name)]
            selection_rows = selection[slice_mask(data, selection, name)]
            evaluations[name] = {
                "fit": warm_report(fit_cost[slice_mask(data, fit, name)], data, fit_rows),
                "inner": warm_report(selection_cost[slice_mask(data, selection, name)], data, selection_rows),
                "actor_warm_movement_sigma_rms": distribution(np.sqrt(np.mean(np.square((selection_action[slice_mask(data, selection, name)] - data["warm_knots"][selection_rows]) / sigma), axis=(1, 2)))),
            }
        fresh_reports = []
        saved_lists = {name: [] for name in ("row", "slice", "center_code", "radius", "center", "action", "raw_action", "cost", "clipped", "critic_log_cost", "fd_true_gradient_z", "fd_predicted_gradient_z")}
        for slice_name in slices:
            local_rows = selection[slice_mask(data, selection, slice_name)]
            local_actor = selection_action[slice_mask(data, selection, slice_name)]
            for position, row in enumerate(local_rows):
                for center_code, center_name, center in ((0, "actor", local_actor[position]), (1, "warm", data["warm_knots"][row])):
                    for radius in config["fresh_local_response"]["one_sided_radii_sigma"]:
                        report, arrays = fresh_bank_record(
                            controller=controller, data=data, row=int(row), center=center,
                            center_name=center_name, radius=float(radius),
                            basis=bases[int(config["fresh_local_response"]["basis_index"])],
                            sigma=sigma, weights=weights, config=config,
                            critics=critics, trainings=trainings, inputs=inputs, device=device,
                        )
                        report["slice"] = slice_name
                        fresh_reports.append(report)
                        saved_lists["row"].append(np.asarray(row, np.int64))
                        saved_lists["slice"].append(np.asarray(slice_name))
                        saved_lists["center_code"].append(np.asarray(center_code, np.int8))
                        saved_lists["radius"].append(np.asarray(radius, np.float32))
                        saved_lists["center"].append(center)
                        for name, value in arrays.items():
                            saved_lists[name].append(value)
        output_arrays = output / f"seed_{seed}.npz"
        np.savez_compressed(
            output_arrays,
            fit_indices=fit,
            selection_indices=selection,
            oof_indices_unevaluated=oof,
            selected_fit_action=fit_action,
            selected_fit_cost=fit_cost,
            selected_inner_action=selection_action,
            selected_inner_cost=selection_cost,
            parameter_names=np.asarray(parameter_names),
            parameter_offsets=parameter_offsets,
            global_actor_parameter_gradient=global_gradient,
            complement_actor_parameter_gradient=complement_gradient,
            stratum_names=np.asarray(all_strata),
            stratum_actor_parameter_gradient=np.stack([stratum_gradients[name] for name in all_strata]),
            selected_actor_encoder_feature=encoder_feature,
            **{f"fresh_{name}": np.stack(value) for name, value in saved_lists.items()},
        )
        record = {
            "seed": seed,
            "selected_round": int(checkpoint["selected_round"]),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": base.sha256(checkpoint_path),
            "source_arrays": str(source_arrays_path),
            "source_arrays_sha256": base.sha256(source_arrays_path),
            "output_arrays": str(output_arrays),
            "output_arrays_sha256": base.sha256(output_arrays),
            "visit_count_by_stratum_through_selected": visit_count_by_stratum,
            "selected_actor_evaluation": evaluations,
            "selected_critic_online_replay": online_reports,
            "actor_gradient_conflict": gradient_report,
            "input_neighborhood": input_neighborhood,
            "fresh_local_response": fresh_reports,
            "new_query_rollouts": int(len(fit) + len(selection) + len(fresh_reports) * int(config["fresh_local_response"]["candidates_per_bank"])),
            "outer_fold_evaluated": False,
        }
        records.append(record)
        print(f"seed={seed} selected={checkpoint['selected_round']} target_grad_cos={gradient_report['target_vs_complement_cosine']:+.4f} fresh_banks={len(fresh_reports)}", flush=True)

    pooled_fresh = {}
    for slice_name in slices:
        pooled_fresh[slice_name] = {}
        for center in config["fresh_local_response"]["centers"]:
            pooled_fresh[slice_name][center] = {}
            for radius in config["fresh_local_response"]["one_sided_radii_sigma"]:
                local = [item for record in records for item in record["fresh_local_response"] if item["slice"] == slice_name and item["center"] == center and abs(item["radius_sigma"] - float(radius)) < 1e-12]
                pooled_fresh[slice_name][center][str(float(radius))] = {
                    "count": len(local),
                    "best_gain_vs_center": distribution(np.asarray([item["best_gain_vs_center"] for item in local])),
                    "best_gain_vs_warm": distribution(np.asarray([item["best_gain_vs_warm"] for item in local])),
                    "best_beats_or_ties_center_fraction": float(np.mean([item["best_gain_vs_center"] >= -1e-7 for item in local])),
                    "best_beats_or_ties_warm_fraction": float(np.mean([item["best_gain_vs_warm"] >= -1e-7 for item in local])),
                    "critic_log_cost_pearson": distribution(np.asarray([item["critic_log_cost_pearson"] for item in local])),
                    "critic_center_relative_sign_accuracy": distribution(np.asarray([item["critic_center_relative_sign_accuracy"] for item in local])),
                    "critic_bank_gain_recovery": distribution(np.asarray([item["critic_bank_gain_recovery"] for item in local])),
                    "fd_gradient_cosine": distribution(np.asarray([item["fd_gradient_cosine"] for item in local])),
                    "fd_gradient_norm_ratio": distribution(np.asarray([item["fd_gradient_norm_ratio"] for item in local])),
                }
    target = config["target_slice"]
    target_gradient_cosines = [record["actor_gradient_conflict"]["target_vs_complement_cosine"] for record in records]
    target_visits = np.asarray([record["visit_count_by_stratum_through_selected"][target] for record in records], np.float64)
    median_visits = np.asarray([np.median(list(record["visit_count_by_stratum_through_selected"].values())) for record in records])
    actor_local = pooled_fresh[target]["actor"]["0.1"]
    target_fd = actor_local["fd_gradient_cosine"]
    control_fd_median = float(np.median([pooled_fresh[name]["actor"]["0.1"]["fd_gradient_cosine"]["median"] for name in config["control_slices"]]))
    raw_reference_medians = {
        name: records[0]["input_neighborhood"][name]["normalized_actor_inputs"]["reference"]["nearest_fit_rms"]["median"]
        for name in all_strata
    }
    target_reference_rank = 1 + sum(
        value > raw_reference_medians[target] for value in raw_reference_medians.values()
    )
    encoder_feature_ranks = []
    for record in records:
        medians = {
            name: record["input_neighborhood"][name]["selected_actor_encoder_feature_global_fit_standardized"]["nearest_fit_rms"]["median"]
            for name in all_strata
        }
        encoder_feature_ranks.append(1 + sum(value > medians[target] for value in medians.values()))
    target_rows = selection[slice_mask(data, selection, target)]
    row_reports = []
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    ordered_seeds = [int(value) for value in config["seeds"]]
    for position, row in enumerate(target_rows):
        actions = [selected_inner_action_by_seed[seed][slice_mask(data, selection, target)][position] for seed in ordered_seeds]
        costs = [float(selected_inner_cost_by_seed[seed][slice_mask(data, selection, target)][position]) for seed in ordered_seeds]
        warm = data["warm_knots"][row]
        warm_cost = float(data["warm_cost"][row])
        pair_distance = [
            float(np.sqrt(np.mean(np.square((actions[left] - actions[right]) / sigma))))
            for left, right in pair_indices
        ]
        movement = [
            float(np.sqrt(np.mean(np.square((action - warm) / sigma))))
            for action in actions
        ]
        regression = [max(cost - warm_cost, 0.0) for cost in costs]
        row_reports.append({
            "row": int(row),
            "episode_id": str(data["episode_id"][row]),
            "control_step": int(data["control_step"][row]),
            "warm_cost": warm_cost,
            "selected_actor_cost_by_seed": dict(zip(map(str, ordered_seeds), costs)),
            "selected_actor_regression_magnitude_by_seed": dict(zip(map(str, ordered_seeds), regression)),
            "actor_warm_movement_sigma_rms_by_seed": dict(zip(map(str, ordered_seeds), movement)),
            "cross_seed_actor_pair_distance_sigma_rms": distribution(np.asarray(pair_distance)),
        })
    worst_row = max(row_reports, key=lambda item: np.median(list(item["selected_actor_regression_magnitude_by_seed"].values())))
    total_regression_by_seed = {
        str(seed): float(sum(item["selected_actor_regression_magnitude_by_seed"][str(seed)] for item in row_reports))
        for seed in ordered_seeds
    }
    worst_row_fraction = {
        str(seed): worst_row["selected_actor_regression_magnitude_by_seed"][str(seed)] / max(total_regression_by_seed[str(seed)], 1e-12)
        for seed in ordered_seeds
    }
    target_episode_tail = {
        "rows": row_reports,
        "worst_row": worst_row,
        "total_regression_magnitude_by_seed": total_regression_by_seed,
        "worst_row_fraction_of_target_regression_by_seed": worst_row_fraction,
    }
    findings = {
        "data_count_balanced": all(len(set(values.values())) == 1 for values in fold_counts.values()),
        "target_online_visitation_ratio_to_stratum_median": distribution(target_visits / median_visits),
        "target_has_actor_local_query_headroom": actor_local["best_gain_vs_center"]["median"] > 0.0,
        "target_critic_fd_cosine_materially_below_controls": target_fd["median"] < control_fd_median - 0.1,
        "target_shared_actor_gradient_conflict_seed_count": int(np.sum(np.asarray(target_gradient_cosines) < 0.0)),
        "target_shared_actor_gradient_conflict_present": int(np.sum(np.asarray(target_gradient_cosines) < 0.0)) >= 2,
        "target_normalized_reference_nearest_fit_distance_rank_descending": int(target_reference_rank),
        "stratum_count_for_input_distance_rank": len(all_strata),
        "target_selected_actor_encoder_feature_distance_rank_descending_by_seed": encoder_feature_ranks,
        "target_input_distribution_shift_present": target_reference_rank <= 2 and max(encoder_feature_ranks) <= 4,
        "target_worst_inner_row": int(worst_row["row"]),
        "target_worst_inner_control_step": int(worst_row["control_step"]),
        "target_worst_row_dominates_regression_all_seeds": min(worst_row_fraction.values()) > 0.5,
    }
    summary = {
        "qualification": "QUERY_OAC_FIXED_LR1E5_HARD_SLICE_DIAGNOSTIC_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "deterministic_runtime_contract": deterministic,
        "fold_stratum_counts": fold_counts,
        "offline_candidate_coverage_fit": coverage,
        "records": records,
        "pooled_fresh_local_response": pooled_fresh,
        "normalized_reference_nearest_fit_distance_median_by_stratum": raw_reference_medians,
        "target_episode_tail": target_episode_tail,
        "findings": findings,
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    base.dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-oac-fixed-lr1e5-hard-slice-diagnostic-v1",
        "qualification": summary["qualification"],
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "source": str(source_dir),
        "source_manifest_sha256": base.sha256(source_dir / "manifest.json"),
        "source_summary_sha256": base.sha256(source_dir / "summary.json"),
        "source_validation_sha256": base.sha256(source_dir / "validation.json"),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(summary_path),
        "run_artifacts": {f"seed{record['seed']}": {"output_arrays_sha256": record["output_arrays_sha256"]} for record in records},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "deterministic_runtime_contract": deterministic,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({"findings": findings, "new_query_rollouts": summary["new_query_rollouts"], "target_fresh": pooled_fresh[target], "target_gradient_cosines": target_gradient_cosines}, indent=2))


if __name__ == "__main__":
    main()
