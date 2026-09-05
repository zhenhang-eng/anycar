#!/usr/bin/env python3
"""Diagnose Query Actor speed-group and empirical bank update geometry."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_actor_aggregation_microstep import objective_weights, vector_cosine  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    distribution,
    load_inputs,
    metrics,
    physical_critic_value,
    sha256,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_update_geometry_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def flatten_gradient(actor: torch.nn.Module) -> tuple[np.ndarray, list[str], np.ndarray]:
    names, offsets, chunks = [], [0], []
    for name, parameter in actor.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing Actor gradient for {name}")
        names.append(name)
        chunks.append(parameter.grad.detach().reshape(-1).cpu().numpy().astype(np.float32))
        offsets.append(offsets[-1] + parameter.numel())
    return np.concatenate(chunks), names, np.asarray(offsets, np.int64)


def predict_conservative(
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    actor_inputs: tuple[np.ndarray, ...],
    critic_inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(len(rows), np.float32)
    actor.eval()
    for critic in critics:
        critic.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            action = torch.from_numpy(actor_predict(actor, actor_inputs, local, device)).to(device)
            values = [
                physical_critic_value(critic, training, critic_inputs, local, action, device)
                for critic, training in zip(critics, trainings)
            ]
            output[start : start + len(local)] = torch.maximum(values[0], values[1]).cpu().numpy()
    return output


def critic_update_direction(
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    actor_inputs: tuple[np.ndarray, ...],
    critic_inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    batch_size = int(config["gradient_contract"]["batch_size"])
    conservative_log = predict_conservative(
        actor, critics, trainings, actor_inputs, critic_inputs, rows, device, batch_size
    )
    weights = objective_weights(
        conservative_log,
        float(config["gradient_contract"]["gamma"]),
        float(config["gradient_contract"]["weight_maximum"]),
    )
    actor.zero_grad(set_to_none=True)
    actor.eval()
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        tensors = tuple(torch.from_numpy(value[local]).to(device) for value in actor_inputs)
        action = actor(*tensors)[1]
        values = [
            physical_critic_value(critic, training, critic_inputs, local, action, device)
            for critic, training in zip(critics, trainings)
        ]
        conservative = torch.maximum(values[0], values[1])
        local_weight = torch.from_numpy(weights[start : start + len(local)]).to(device)
        (local_weight * conservative).sum().div(len(rows)).backward()
    gradient, names, offsets = flatten_gradient(actor)
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    return -gradient, names, offsets, weights


def empirical_bank_response(
    arrays: dict[str, np.ndarray], fit: np.ndarray, config: dict
) -> dict[str, np.ndarray]:
    width = int(config["population"]["candidate_count"])
    groups = int(config["population"]["expected_online_groups_per_seed"])
    count = groups * width
    for name in ("online_state_index", "online_action", "online_cost", "online_group", "online_role"):
        if len(arrays[name]) != count:
            raise AssertionError(f"unexpected {name} length")
    group_id = arrays["online_group"].reshape(groups, width)
    if not np.all(group_id == np.arange(groups)[:, None]):
        raise AssertionError("online group layout changed")
    rows_by_group = arrays["online_state_index"].reshape(groups, width)
    if not np.all(rows_by_group == rows_by_group[:, :1]):
        raise AssertionError("one online group spans multiple states")
    roles = arrays["online_role"].reshape(groups, width)
    if not np.all(roles[:, 0] == "actor"):
        raise AssertionError("candidate zero is not the saved Actor center")
    rows = rows_by_group[:, 0].astype(np.int64)
    if not np.all(np.isin(rows, fit)):
        raise AssertionError("online bank consumed a non-fit row")
    actions = arrays["online_action"].reshape(groups, width, 8, 2)
    costs = arrays["online_cost"].reshape(groups, width).astype(np.float64)
    best_index = np.argmin(costs, axis=1)
    best = actions[np.arange(groups), best_index]
    center = actions[:, 0]
    sigma = np.asarray(config["gradient_contract"]["noise_sigma"], np.float64).reshape(1, 1, 2)
    displacement = ((best.astype(np.float64) - center.astype(np.float64)) / sigma).reshape(groups, -1)
    gain = costs[:, 0] - costs[np.arange(groups), best_index]
    if gain.min() < -1e-8:
        raise AssertionError("bank best is worse than its retained center")
    denominator = np.square(displacement).sum(axis=1) + float(config["gradient_contract"]["secant_epsilon"])
    secant = gain[:, None] * displacement / denominator[:, None]
    unique_rows = np.unique(rows)
    mean_displacement = np.stack([displacement[rows == row].mean(axis=0) for row in unique_rows])
    mean_secant = np.stack([secant[rows == row].mean(axis=0) for row in unique_rows])
    return {
        "rows": unique_rows.astype(np.int64),
        "displacement": mean_displacement.reshape(-1, 8, 2).astype(np.float32),
        "secant": mean_secant.reshape(-1, 8, 2).astype(np.float32),
        "group_rows": rows,
        "group_best_index": best_index.astype(np.int16),
        "group_gain": gain.astype(np.float32),
        "group_displacement_norm": np.linalg.norm(displacement, axis=1).astype(np.float32),
    }


def empirical_update_direction(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    actor.zero_grad(set_to_none=True)
    actor.eval()
    sigma_tensor = torch.from_numpy(sigma.astype(np.float32)).to(device).reshape(1, 1, 2)
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        tensors = tuple(torch.from_numpy(value[local]).to(device) for value in actor_inputs)
        action = actor(*tensors)[1] / sigma_tensor
        local_response = torch.from_numpy(response[start : start + len(local)]).to(device)
        (action * local_response).sum().div(len(rows)).backward()
    return flatten_gradient(actor)


def geometry_report(aggregate: np.ndarray, grouped: np.ndarray, speeds: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray([
        [vector_cosine(grouped[left], grouped[right]) for right in range(len(speeds))]
        for left in range(len(speeds))
    ], np.float64)
    off = matrix[np.triu_indices(len(speeds), 1)]
    return {
        "aggregate_norm": float(np.linalg.norm(aggregate.astype(np.float64))),
        "speed_norm": {str(int(speed)): float(np.linalg.norm(grouped[i].astype(np.float64))) for i, speed in enumerate(speeds)},
        "aggregate_to_speed_cosine": {str(int(speed)): vector_cosine(aggregate, grouped[i]) for i, speed in enumerate(speeds)},
        "speed_pair_cosine_matrix": matrix.tolist(),
        "speed_pair_negative_fraction": float(np.mean(off < 0.0)),
        "speed_pair_material_negative_fraction": float(np.mean(off <= -0.05)),
        "speed_pair_cosine_minimum": float(off.min()),
        "speed_pair_cosine_median": float(np.median(off)),
    }


def state_from_update(
    base_state: dict[str, torch.Tensor], names: list[str], offsets: np.ndarray,
    update: np.ndarray, multiplier: float,
) -> dict[str, torch.Tensor]:
    output = copy.deepcopy(base_state)
    for index, name in enumerate(names):
        local = torch.from_numpy(update[offsets[index] : offsets[index + 1]].reshape(base_state[name].shape))
        output[name] = base_state[name] + float(multiplier) * local.to(base_state[name].device)
    return output


def output_rms(
    actor: torch.nn.Module, base: np.ndarray, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
    sigma: np.ndarray, device: torch.device,
) -> float:
    current = actor_predict(actor, inputs, rows, device)
    return float(np.sqrt(np.mean(np.square((current - base) / sigma))))


def calibrated_update(
    actor: torch.nn.Module, base_state: dict[str, torch.Tensor], base_output: np.ndarray,
    inputs: tuple[np.ndarray, ...], rows: np.ndarray, sigma: np.ndarray,
    update: np.ndarray, names: list[str], offsets: np.ndarray, target: float,
    device: torch.device,
) -> tuple[float, float]:
    norm = float(np.linalg.norm(update.astype(np.float64)))
    if norm <= 1e-20:
        raise AssertionError("zero parameter update direction")
    direction = update / norm
    lower, upper = 0.0, 1e-7
    while True:
        actor.load_state_dict(state_from_update(base_state, names, offsets, direction, upper), strict=True)
        if output_rms(actor, base_output, inputs, rows, sigma, device) >= target:
            break
        upper *= 2.0
        if upper > 10.0:
            raise AssertionError("failed to bracket output step")
    for _ in range(32):
        midpoint = (lower + upper) / 2.0
        actor.load_state_dict(state_from_update(base_state, names, offsets, direction, midpoint), strict=True)
        if output_rms(actor, base_output, inputs, rows, sigma, device) < target:
            lower = midpoint
        else:
            upper = midpoint
    multiplier = (lower + upper) / 2.0
    actor.load_state_dict(state_from_update(base_state, names, offsets, direction, multiplier), strict=True)
    return multiplier, output_rms(actor, base_output, inputs, rows, sigma, device)


def analyze_seed(
    seed: int, record: dict, data: dict[str, np.ndarray], controller: TorchMPPIController,
    config: dict, device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    checkpoint_path, arrays_path = Path(record["checkpoint"]), Path(record["arrays"])
    if sha256(checkpoint_path) != record["checkpoint_sha256"] or sha256(arrays_path) != record["arrays_sha256"]:
        raise AssertionError(f"seed {seed} source artifact hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with np.load(arrays_path, allow_pickle=False) as archive:
        source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    fit = checkpoint["fit_indices"].astype(np.int64)
    selection = checkpoint["selection_indices"].astype(np.int64)
    outer = checkpoint["outer_indices_unevaluated"].astype(np.int64)
    expected = tuple(config["split_contract"]["expected_sizes"])
    if (len(fit), len(selection), len(outer)) != expected:
        raise AssertionError("split sizes changed")
    actor_inputs = load_inputs(data, checkpoint["actor_normalization"])
    critic_inputs = load_inputs(data, checkpoint["critic_normalization"])
    actor = actor_from_payload(checkpoint, "selected_actor_state_dict", device)
    critics, trainings = [], []
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
        critic.eval()
        critics.append(critic)
        trainings.append(checkpoint[f"critic{twin}_training"])
    sigma = np.asarray(config["gradient_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    speeds = np.asarray(config["gradient_contract"]["speed_values_kph"], np.int64)
    batch_size = int(config["gradient_contract"]["batch_size"])
    empirical = empirical_bank_response(source_arrays, fit, config)
    visited = empirical["rows"]

    critic_full, names, offsets, full_weights = critic_update_direction(
        actor, critics, trainings, actor_inputs, critic_inputs, fit, config, device
    )
    critic_visited, local_names, local_offsets, visited_weights = critic_update_direction(
        actor, critics, trainings, actor_inputs, critic_inputs, visited, config, device
    )
    if names != local_names or not np.array_equal(offsets, local_offsets):
        raise AssertionError("Actor parameter layout changed")
    critic_speed, critic_visited_speed = [], []
    for speed in speeds:
        full_rows = fit[data["speed_kph"][fit] == speed]
        visit_rows = visited[data["speed_kph"][visited] == speed]
        critic_speed.append(critic_update_direction(
            actor, critics, trainings, actor_inputs, critic_inputs, full_rows, config, device
        )[0])
        critic_visited_speed.append(critic_update_direction(
            actor, critics, trainings, actor_inputs, critic_inputs, visit_rows, config, device
        )[0])
    critic_speed = np.stack(critic_speed)
    critic_visited_speed = np.stack(critic_visited_speed)

    empirical_secant, empirical_raw = [], []
    for response_name in ("secant", "displacement"):
        aggregate, response_names, response_offsets = empirical_update_direction(
            actor, actor_inputs, visited, empirical[response_name], sigma, device, batch_size
        )
        if response_names != names or not np.array_equal(response_offsets, offsets):
            raise AssertionError("empirical parameter layout changed")
        grouped = []
        for speed in speeds:
            mask = data["speed_kph"][visited] == speed
            grouped.append(empirical_update_direction(
                actor, actor_inputs, visited[mask], empirical[response_name][mask], sigma, device, batch_size
            )[0])
        if response_name == "secant":
            empirical_secant = [aggregate, np.stack(grouped)]
        else:
            empirical_raw = [aggregate, np.stack(grouped)]

    base_state = copy.deepcopy(actor.state_dict())
    base_fit_action = actor_predict(actor, actor_inputs, fit, device)
    base_selection_action = actor_predict(actor, actor_inputs, selection, device)
    selected_round = int(checkpoint["selected_round"])
    stored_action = source_arrays["selection_round_action"][selected_round].astype(np.float32)
    base_selection_cost = source_arrays["selection_round_cost"][selected_round].astype(np.float32)
    if np.max(np.abs(base_selection_action - stored_action)) > 1e-6:
        raise AssertionError("selected Actor action does not match saved round")
    direction_names = ["critic_gamma1", "empirical_secant"]
    directions = [critic_full, empirical_secant[0]]
    step_actions, step_costs, step_multiplier, step_achieved, step_reports = [], [], [], [], {}
    target = float(config["microstep_contract"]["target_full_fit_output_sigma_rms"])
    weights_cost = {name: float(value) for name, value in config["cost_weights"].items()}
    for direction_name, direction in zip(direction_names, directions):
        actor.load_state_dict(base_state, strict=True)
        multiplier, achieved = calibrated_update(
            actor, base_state, base_fit_action, actor_inputs, fit, sigma,
            direction, names, offsets, target, device,
        )
        action = actor_predict(actor, actor_inputs, selection, device)
        cost = batched_direct_cost(controller, data, selection, action, weights_cost)
        step_actions.append(action)
        step_costs.append(cost)
        step_multiplier.append(multiplier)
        step_achieved.append(achieved)
        step_reports[direction_name] = metrics(
            cost, base_selection_cost, data["warm_cost"][selection], data["speed_kph"][selection]
        )
    actor.load_state_dict(base_state, strict=True)

    critic_empirical_speed = {
        str(int(speed)): vector_cosine(critic_visited_speed[index], empirical_secant[1][index])
        for index, speed in enumerate(speeds)
    }
    seed_report = {
        "seed": seed,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": record["checkpoint_sha256"],
        "source_arrays": str(arrays_path),
        "source_arrays_sha256": record["arrays_sha256"],
        "selected_round": selected_round,
        "visited_state_count": int(len(visited)),
        "visit_count": int(len(empirical["group_rows"])),
        "bank_gain": distribution(empirical["group_gain"]),
        "bank_displacement_sigma_norm": distribution(empirical["group_displacement_norm"]),
        "gamma1_weight": {"full_fit": distribution(full_weights), "visited": distribution(visited_weights)},
        "critic_full_geometry": geometry_report(critic_full, critic_speed, speeds),
        "critic_visited_geometry": geometry_report(critic_visited, critic_visited_speed, speeds),
        "empirical_secant_geometry": geometry_report(empirical_secant[0], empirical_secant[1], speeds),
        "empirical_raw_geometry": geometry_report(empirical_raw[0], empirical_raw[1], speeds),
        "critic_vs_empirical_secant_cosine": vector_cosine(critic_visited, empirical_secant[0]),
        "critic_vs_empirical_raw_cosine": vector_cosine(critic_visited, empirical_raw[0]),
        "critic_vs_empirical_secant_speed_cosine": critic_empirical_speed,
        "microsteps": step_reports,
        "oof_evaluated": False,
        "new_query_rollouts": int(len(selection) * len(direction_names)),
    }
    result_arrays = {
        "fit_indices": fit, "selection_indices": selection, "outer_indices_unevaluated": outer,
        "visited_indices": visited, "speed_values": speeds,
        "parameter_names": np.asarray(names), "parameter_offsets": offsets,
        "group_rows": empirical["group_rows"], "group_best_index": empirical["group_best_index"],
        "group_gain": empirical["group_gain"],
        "group_displacement_norm": empirical["group_displacement_norm"],
        "state_mean_displacement": empirical["displacement"], "state_mean_secant": empirical["secant"],
        "critic_full_update": critic_full, "critic_visited_update": critic_visited,
        "critic_speed_update": critic_speed, "critic_visited_speed_update": critic_visited_speed,
        "empirical_secant_update": empirical_secant[0], "empirical_secant_speed_update": empirical_secant[1],
        "empirical_raw_update": empirical_raw[0], "empirical_raw_speed_update": empirical_raw[1],
        "base_fit_action": base_fit_action, "base_selection_action": base_selection_action,
        "base_selection_cost": base_selection_cost, "direction_names": np.asarray(direction_names),
        "step_parameter_multiplier": np.asarray(step_multiplier, np.float64),
        "step_achieved_sigma_rms": np.asarray(step_achieved, np.float64),
        "step_selection_action": np.stack(step_actions), "step_selection_cost": np.stack(step_costs),
    }
    return seed_report, result_arrays


def route(records: list[dict[str, Any]], config: dict, arrays: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    speeds = np.asarray(config["gradient_contract"]["speed_values_kph"], np.int64)
    threshold = float(config["decision_gate"]["material_negative_cosine"])
    required = int(config["decision_gate"]["stable_pair_minimum_seed_count"])
    stable_pairs = []
    pair_reports = {}
    for left in range(len(speeds)):
        for right in range(left + 1, len(speeds)):
            values = [record["critic_full_geometry"]["speed_pair_cosine_matrix"][left][right] for record in records]
            count = int(np.sum(np.asarray(values) <= threshold))
            name = f"{int(speeds[left])}-{int(speeds[right])}"
            pair_reports[name] = {"cosine_by_seed": values, "material_negative_seed_count": count}
            if count >= required:
                stable_pairs.append(name)
    aligned_count = int(np.sum(np.asarray([
        record["critic_vs_empirical_secant_cosine"] for record in records
    ]) > 0.0))
    pooled_critic_gain = np.concatenate([
        item["base_selection_cost"] - item["step_selection_cost"][0] for item in arrays
    ]).astype(np.float64)
    pooled_empirical_gain = np.concatenate([
        item["base_selection_cost"] - item["step_selection_cost"][1] for item in arrays
    ]).astype(np.float64)
    pcgrad_conflict = len(stable_pairs) >= int(config["decision_gate"]["stable_pair_minimum_count_for_pcgrad"])
    critic_aligned = aligned_count >= int(config["decision_gate"]["critic_empirical_positive_minimum_seed_count"])
    critic_step_ok = float(pooled_critic_gain.mean()) >= float(config["decision_gate"]["critic_step_pooled_mean_gain_minimum"])
    if not critic_aligned:
        decision = "CRITIC_EMPIRICAL_DIRECTION_MISALIGNED_DO_NOT_PCGRAD"
    elif pcgrad_conflict and critic_step_ok:
        decision = "ADVANCE_SPEED_PCGRAD_CAGRAD_PILOT"
    elif pcgrad_conflict:
        decision = "SPEED_CONFLICT_PRESENT_BUT_CRITIC_STEP_FAILS_DO_NOT_PCGRAD"
    else:
        decision = "NO_STABLE_SPEED_CONFLICT_DO_NOT_PCGRAD"
    return {
        "decision": decision,
        "stable_speed_pairs": stable_pairs,
        "stable_speed_pair_count": len(stable_pairs),
        "pair_reports": pair_reports,
        "critic_empirical_positive_seed_count": aligned_count,
        "critic_step_pooled_gain": distribution(pooled_critic_gain),
        "empirical_step_pooled_gain": distribution(pooled_empirical_gain),
        "checks": {
            "minimum_two_stable_pairs": pcgrad_conflict,
            "critic_empirical_positive_at_least_two_seeds": critic_aligned,
            "critic_step_pooled_mean_nonnegative": critic_step_ok,
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed boundary violation")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    source = Path(config["sources"]["coarse77_three_seed"])
    source_summary_path, source_validation_path = source / "summary.json", source / "validation.json"
    if sha256(source_summary_path) != config["sources"]["coarse77_summary_sha256"]:
        raise AssertionError("source summary hash mismatch")
    if sha256(source_validation_path) != config["sources"]["coarse77_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["coarse77_qualification"]:
        raise AssertionError("source qualification changed")
    source_summary = json.loads(source_summary_path.read_text())
    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != tuple(config["split_contract"]["expected_sizes"]):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    output.mkdir(parents=True)
    records, result_arrays, hashes = [], [], {}
    for seed in config["population"]["seeds"]:
        record = source_summary["records"]["coarse_to_fine77"][str(seed)]
        report, arrays = analyze_seed(int(seed), record, data, controller, config, device)
        arrays_path = output / f"seed_{seed}.npz"
        np.savez_compressed(arrays_path, **arrays)
        report["output_arrays"] = str(arrays_path)
        report["output_arrays_sha256"] = sha256(arrays_path)
        hashes[str(seed)] = report["output_arrays_sha256"]
        records.append(report)
        result_arrays.append(arrays)
        print(
            f"seed={seed} critic_empirical_cos={report['critic_vs_empirical_secant_cosine']:.4f} "
            f"critic_step_gain={report['microsteps']['critic_gamma1']['gain_vs_round0']['mean']:.6f} "
            f"empirical_step_gain={report['microsteps']['empirical_secant']['gain_vs_round0']['mean']:.6f}",
            flush=True,
        )
    routing = route(records, config, result_arrays)
    summary = {
        "qualification": "QUERY_ACTOR_UPDATE_GEOMETRY_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": records, "routing": routing, "decision": routing["decision"],
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-actor-update-geometry-v1", "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "source_summary": str(source_summary_path), "source_summary_sha256": sha256(source_summary_path),
        "source_validation": str(source_validation_path), "source_validation_sha256": sha256(source_validation_path),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "seed_arrays_sha256": hashes, "summary_sha256": sha256(output / "summary.json"),
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(routing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
