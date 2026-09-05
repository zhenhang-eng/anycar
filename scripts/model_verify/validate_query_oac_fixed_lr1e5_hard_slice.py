#!/usr/bin/env python3
"""Independently validate the fixed-LR Query OAC hard-slice diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import run_query_oac_gamma1_k_scan as base
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_actor_aggregation_microstep import (
    actor_gradient,
    objective_weights,
    predict_conservative_log,
    vector_cosine,
)
from run_query_single_center_oac20to1 import (
    actor_from_payload,
    actor_predict,
    direct_cost,
    distribution,
    load_inputs,
    physical_critic_value,
    response_bank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_fixed_lr1e5_hard_slice_diagnostic_20260903_v1"


def deterministic_contract() -> dict[str, object]:
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
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def max_error(left: np.ndarray | float, right: np.ndarray | float) -> float:
    left64 = np.asarray(left, np.float64)
    right64 = np.asarray(right, np.float64)
    return 0.0 if left64.size == 0 else float(np.max(np.abs(left64 - right64)))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if len(left) < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def slice_mask(data: dict[str, np.ndarray], rows: np.ndarray, name: str) -> np.ndarray:
    speed, variant = (int(value) for value in name.split(":"))
    return (data["speed_kph"][rows] == speed) & (data["variant_index"][rows] == variant)


def encoder_features(
    actor: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(inputs[0]), batch_size):
            tensors = tuple(torch.from_numpy(value[start : start + batch_size]).to(device) for value in inputs)
            output.append(
                actor.encoder(
                    tensors[0], tensors[1], tensors[2],
                    torch.zeros_like(tensors[3]), torch.zeros_like(tensors[4]),
                    torch.zeros_like(tensors[5]),
                ).cpu().numpy()
            )
    return np.concatenate(output).astype(np.float32)


def nearest_rms(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, np.float32).reshape(len(query), -1)
    reference = np.asarray(reference, np.float32).reshape(len(reference), -1)
    distance = np.sqrt(np.mean(np.square(query[:, None] - reference[None]), axis=2))
    return distance.min(axis=1).astype(np.float32)


def critic_values(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    row: int,
    action: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rows = np.full(len(action), row, np.int64)
    tensor = torch.from_numpy(action).to(device)
    with torch.no_grad():
        values = [
            physical_critic_value(critic, training, inputs, rows, tensor, device).cpu().numpy()
            for critic, training in zip(critics, trainings)
        ]
    return np.maximum(values[0], values[1]).astype(np.float32)


def critic_gradient(
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
    gradient = torch.autograd.grad(torch.maximum(values[0], values[1]).sum(), action)[0]
    return (gradient * torch.from_numpy(sigma).to(device)).detach().cpu().numpy().reshape(16).astype(np.float32)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path, summary_path = output / "manifest.json", output / "summary.json"
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    deterministic = deterministic_contract()
    replay_dir = Path(manifest["absolute_replay"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    data = load_npz(replay_dir / "replay.npz")
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = base.QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = base.TorchMPPIController(
        base.TorchQueryRolloutBackend(query),
        base.TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["noise_sigma"], np.float32).reshape(1, 2)
    bases = base.basis_bank()
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    oof = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    strata = [
        f"{int(speed)}:{int(variant)}"
        for speed in sorted(np.unique(data["speed_kph"][fit]).tolist())
        for variant in sorted(np.unique(data["variant_index"][fit]).tolist())
    ]
    checks = {
        "config_hash": base.sha256(config_path) == manifest["config_sha256"],
        "runner_hash": base.sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": base.sha256(summary_path) == manifest["summary_sha256"],
        "source_manifest_hash": base.sha256(Path(manifest["source"]) / "manifest.json") == manifest["source_manifest_sha256"],
        "source_summary_hash": base.sha256(Path(manifest["source"]) / "summary.json") == manifest["source_summary_sha256"],
        "source_validation_hash": base.sha256(Path(manifest["source"]) / "validation.json") == manifest["source_validation_sha256"],
        "replay_hash": base.sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "query_checkpoint_hash": base.sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
        "episode_disjoint": not bool(
            set(data["episode_id"][fit]) & set(data["episode_id"][selection])
            or set(data["episode_id"][fit]) & set(data["episode_id"][oof])
            or set(data["episode_id"][selection]) & set(data["episode_id"][oof])
        ),
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"]) and not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_absent": not bool(summary["dbm_fields_or_labels_consumed"]) and not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(summary["query_analytic_gradient_consumed"]) and not bool(manifest["query_analytic_gradient_consumed"]),
        "outer_unevaluated": not bool(summary["outer_fold_evaluated"]) and not bool(manifest["outer_fold_evaluated"]),
        "balanced_fold_strata": all(len(set(value.values())) == 1 for value in summary["fold_stratum_counts"].values()),
        "deterministic_contract": summary["deterministic_runtime_contract"] == deterministic and manifest["deterministic_runtime_contract"] == deterministic,
    }
    errors = {
        "fit_action": 0.0, "fit_cost": 0.0, "inner_action": 0.0, "inner_cost": 0.0,
        "encoder_feature": 0.0, "global_gradient": 0.0, "complement_gradient": 0.0,
        "stratum_gradient": 0.0, "fresh_action": 0.0, "fresh_raw_action": 0.0,
        "fresh_cost": 0.0, "fresh_critic": 0.0, "fresh_true_gradient": 0.0,
        "fresh_predicted_gradient": 0.0, "fresh_report_metric": 0.0,
        "reference_nn_median": 0.0, "fresh_cost_relative": 0.0,
    }
    selected_actions: dict[int, np.ndarray] = {}
    selected_costs: dict[int, np.ndarray] = {}
    fresh_bank_count = 0
    replayed_candidate_count = 0
    fresh_best_index_identity = True
    worst_fresh_cost_replay: dict[str, object] = {}
    for record in summary["records"]:
        seed = int(record["seed"])
        arrays_path = Path(record["output_arrays"])
        checkpoint_path = Path(record["source_checkpoint"])
        arrays = load_npz(arrays_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checks[f"seed{seed}_arrays_hash"] = base.sha256(arrays_path) == record["output_arrays_sha256"] == manifest["run_artifacts"][f"seed{seed}"]["output_arrays_sha256"]
        checks[f"seed{seed}_checkpoint_hash"] = base.sha256(checkpoint_path) == record["source_checkpoint_sha256"]
        checks[f"seed{seed}_indices"] = bool(np.array_equal(arrays["fit_indices"], fit) and np.array_equal(arrays["selection_indices"], selection) and np.array_equal(arrays["oof_indices_unevaluated"], oof))
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(
            {"actor_training": checkpoint["actor_training"], "selected_actor_state_dict": checkpoint["selected_actor_state_dict"]},
            "selected_actor_state_dict", device,
        )
        fit_action = actor_predict(actor, inputs, fit, device)
        inner_action = actor_predict(actor, inputs, selection, device)
        fit_cost = direct_cost(controller, data, fit, fit_action, weights)
        inner_cost = direct_cost(controller, data, selection, inner_action, weights)
        selected_actions[seed], selected_costs[seed] = inner_action, inner_cost
        errors["fit_action"] = max(errors["fit_action"], max_error(fit_action, arrays["selected_fit_action"]))
        errors["fit_cost"] = max(errors["fit_cost"], max_error(fit_cost, arrays["selected_fit_cost"]))
        errors["inner_action"] = max(errors["inner_action"], max_error(inner_action, arrays["selected_inner_action"]))
        errors["inner_cost"] = max(errors["inner_cost"], max_error(inner_cost, arrays["selected_inner_cost"]))
        feature = encoder_features(actor, inputs, device, int(config["actor_gradient"]["batch_size"]))
        errors["encoder_feature"] = max(errors["encoder_feature"], max_error(feature, arrays["selected_actor_encoder_feature"]))
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        conservative = predict_conservative_log(actor, critics, trainings, inputs, fit, device, int(config["actor_gradient"]["batch_size"]))
        objective = objective_weights(conservative, float(config["actor_gradient"]["gamma"]), float(config["actor_gradient"]["weight_maximum"]))
        global_gradient, names, offsets = actor_gradient(actor, critics, trainings, inputs, fit, objective, device, int(config["actor_gradient"]["batch_size"]))
        target_mask = slice_mask(data, fit, config["target_slice"])
        complement_gradient, _, _ = actor_gradient(actor, critics, trainings, inputs, fit[~target_mask], objective[~target_mask], device, int(config["actor_gradient"]["batch_size"]))
        stratum_gradients = []
        for name in strata:
            mask = slice_mask(data, fit, name)
            gradient, local_names, local_offsets = actor_gradient(actor, critics, trainings, inputs, fit[mask], objective[mask], device, int(config["actor_gradient"]["batch_size"]))
            checks[f"seed{seed}_{name}_gradient_layout"] = names == local_names and np.array_equal(offsets, local_offsets)
            stratum_gradients.append(gradient)
        errors["global_gradient"] = max(errors["global_gradient"], max_error(global_gradient, arrays["global_actor_parameter_gradient"]))
        errors["complement_gradient"] = max(errors["complement_gradient"], max_error(complement_gradient, arrays["complement_actor_parameter_gradient"]))
        errors["stratum_gradient"] = max(errors["stratum_gradient"], max_error(np.stack(stratum_gradients), arrays["stratum_actor_parameter_gradient"]))
        target_index = strata.index(config["target_slice"])
        errors["fresh_report_metric"] = max(
            errors["fresh_report_metric"],
            abs(vector_cosine(stratum_gradients[target_index], complement_gradient) - float(record["actor_gradient_conflict"]["target_vs_complement_cosine"])),
        )
        fit_mean, fit_std = feature[fit].mean(0, keepdims=True), feature[fit].std(0, keepdims=True) + 1e-6
        standardized = (feature - fit_mean) / fit_std
        for name in strata:
            fit_rows, inner_rows = fit[slice_mask(data, fit, name)], selection[slice_mask(data, selection, name)]
            reference_distance = nearest_rms(inputs[1][inner_rows], inputs[1][fit_rows])
            saved_median = record["input_neighborhood"][name]["normalized_actor_inputs"]["reference"]["nearest_fit_rms"]["median"]
            errors["reference_nn_median"] = max(errors["reference_nn_median"], abs(float(np.median(reference_distance)) - float(saved_median)))
            feature_distance = nearest_rms(standardized[inner_rows], standardized[fit_rows])
            saved_feature = record["input_neighborhood"][name]["selected_actor_encoder_feature_global_fit_standardized"]["nearest_fit_rms"]["median"]
            errors["reference_nn_median"] = max(errors["reference_nn_median"], abs(float(np.median(feature_distance)) - float(saved_feature)))
        bank_identity = True
        for index, report in enumerate(record["fresh_local_response"]):
            row = int(arrays["fresh_row"][index])
            radius = float(arrays["fresh_radius"][index])
            center = arrays["fresh_center"][index]
            action, cost, raw, clipped = response_bank(
                controller, data, row, center, radius,
                bases[int(config["fresh_local_response"]["basis_index"])],
                sigma, weights, config,
            )
            predicted = critic_values(critics, trainings, inputs, row, action, device)
            x = ((action[1:33] - center[None]) / sigma).reshape(32, 16).astype(np.float64)
            y = np.log1p(cost[1:33].astype(np.float64)) - np.log1p(float(cost[0]))
            ridge = float(config["fresh_local_response"]["finite_difference_ridge"])
            true_gradient = np.linalg.solve(x.T @ x + ridge * np.eye(16), x.T @ y).astype(np.float32)
            predicted_gradient = critic_gradient(critics, trainings, inputs, row, center, sigma, device)
            errors["fresh_action"] = max(errors["fresh_action"], max_error(action, arrays["fresh_action"][index]))
            errors["fresh_raw_action"] = max(errors["fresh_raw_action"], max_error(raw, arrays["fresh_raw_action"][index]))
            errors["fresh_cost"] = max(errors["fresh_cost"], max_error(cost, arrays["fresh_cost"][index]))
            cost_error = np.abs(cost.astype(np.float64) - arrays["fresh_cost"][index].astype(np.float64))
            relative_error = cost_error / np.maximum(np.abs(arrays["fresh_cost"][index].astype(np.float64)), 1.0)
            errors["fresh_cost_relative"] = max(errors["fresh_cost_relative"], float(np.max(relative_error)))
            local_worst = int(np.argmax(cost_error))
            if float(cost_error[local_worst]) >= float(worst_fresh_cost_replay.get("absolute_error", -1.0)):
                worst_fresh_cost_replay = {
                    "seed": seed, "bank_index": index, "row": row,
                    "slice": str(arrays["fresh_slice"][index]),
                    "center": report["center"], "radius_sigma": radius,
                    "candidate_index": local_worst,
                    "saved_cost": float(arrays["fresh_cost"][index, local_worst]),
                    "replayed_cost": float(cost[local_worst]),
                    "absolute_error": float(cost_error[local_worst]),
                    "relative_error": float(relative_error[local_worst]),
                }
            errors["fresh_critic"] = max(errors["fresh_critic"], max_error(predicted, arrays["fresh_critic_log_cost"][index]))
            errors["fresh_true_gradient"] = max(errors["fresh_true_gradient"], max_error(true_gradient, arrays["fresh_fd_true_gradient_z"][index]))
            errors["fresh_predicted_gradient"] = max(errors["fresh_predicted_gradient"], max_error(predicted_gradient, arrays["fresh_fd_predicted_gradient_z"][index]))
            bank_identity = bank_identity and bool(
                row == int(report["row"])
                and str(arrays["fresh_slice"][index]) == report["slice"]
                and ("actor" if int(arrays["fresh_center_code"][index]) == 0 else "warm") == report["center"]
                and np.array_equal(clipped, arrays["fresh_clipped"][index])
            )
            true_delta = np.log1p(cost[1:].astype(np.float64)) - np.log1p(float(cost[0]))
            predicted_delta = predicted[1:].astype(np.float64) - float(predicted[0])
            material = np.abs(true_delta) > 1e-7
            best, critic_best = int(np.argmin(cost)), int(np.argmin(predicted))
            fresh_best_index_identity = fresh_best_index_identity and bool(
                best == int(np.argmin(arrays["fresh_cost"][index]))
                and critic_best == int(np.argmin(arrays["fresh_critic_log_cost"][index]))
            )
            denominator = max(float(cost[0] - cost[best]), 1e-12)
            recovery = float((cost[0] - cost[critic_best]) / denominator) if cost[0] > cost[best] + 1e-8 else 0.0
            derived = {
                "center_cost": float(cost[0]), "best_cost": float(cost[best]),
                "critic_selected_cost": float(cost[critic_best]),
                "critic_log_cost_pearson": correlation(predicted, np.log1p(cost)),
                "critic_center_relative_sign_accuracy": float(np.mean(np.sign(true_delta[material]) == np.sign(predicted_delta[material]))) if np.any(material) else 1.0,
                "critic_bank_gain_recovery": recovery,
                "fd_gradient_cosine": vector_cosine(true_gradient, predicted_gradient),
                "fd_gradient_norm_ratio": float(np.linalg.norm(predicted_gradient) / max(np.linalg.norm(true_gradient), 1e-12)),
            }
            errors["fresh_report_metric"] = max(errors["fresh_report_metric"], *(abs(value - float(report[name])) for name, value in derived.items()))
            fresh_bank_count += 1
            replayed_candidate_count += len(cost)
        checks[f"seed{seed}_fresh_bank_identity"] = bank_identity
    raw_reference_medians = {}
    first_record = summary["records"][0]
    for name in strata:
        raw_reference_medians[name] = float(first_record["input_neighborhood"][name]["normalized_actor_inputs"]["reference"]["nearest_fit_rms"]["median"])
    target = config["target_slice"]
    reference_rank = 1 + sum(value > raw_reference_medians[target] for value in raw_reference_medians.values())
    target_rows = selection[slice_mask(data, selection, target)]
    regressions = np.stack([
        np.maximum(selected_costs[seed][slice_mask(data, selection, target)] - data["warm_cost"][target_rows], 0.0)
        for seed in map(int, config["seeds"])
    ])
    worst_position = int(np.argmax(np.median(regressions, axis=0)))
    checks.update({
        "fresh_bank_count": fresh_bank_count == len(config["seeds"]) * len([target, *config["control_slices"]]) * 6 * 2 * len(config["fresh_local_response"]["one_sided_radii_sigma"]),
        "query_rollout_count": replayed_candidate_count + len(config["seeds"]) * (len(fit) + len(selection)) == int(summary["new_query_rollouts"]),
        "reference_rank": reference_rank == int(summary["findings"]["target_normalized_reference_nearest_fit_distance_rank_descending"]),
        "worst_target_row": int(target_rows[worst_position]) == int(summary["findings"]["target_worst_inner_row"]),
        "worst_target_step": int(data["control_step"][target_rows[worst_position]]) == int(summary["findings"]["target_worst_inner_control_step"]),
        "fresh_best_indices_stable": fresh_best_index_identity,
    })
    thresholds = {name: 1e-5 for name in errors}
    # Fitted response proposals can differ by one float32 action ULP between
    # otherwise deterministic CUDA executions; Query rollout is locally
    # sensitive to that ULP.  Require the selected minima to remain identical
    # and cap both the absolute and scale-relative cost drift explicitly.
    thresholds["fresh_cost"] = 2e-2
    thresholds["fresh_cost_relative"] = 5e-4
    thresholds["fresh_report_metric"] = 2e-5
    checks["numeric_replay"] = all(errors[name] <= thresholds[name] for name in errors)
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_OAC_FIXED_LR1E5_HARD_SLICE_DIAGNOSTIC_INDEPENDENT_PASS" if passed else "QUERY_OAC_FIXED_LR1E5_HARD_SLICE_DIAGNOSTIC_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": base.sha256(Path(__file__).resolve()),
        "checks": checks,
        "max_absolute_errors": errors,
        "thresholds": thresholds,
        "fresh_banks_replayed": fresh_bank_count,
        "query_candidates_replayed": replayed_candidate_count,
        "worst_fresh_cost_replay": worst_fresh_cost_replay,
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "validation.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
