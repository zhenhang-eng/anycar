#!/usr/bin/env python3
"""Generate T1 high-budget/multi-center DBM proposal-teacher labels.

Each snapshot first runs multi-start CEM searches from the warm, T0-best, and
T0-soft centers.  A short list of resulting centers is then evaluated with the
same online-sized perturbation batches and common random numbers.  Selection is
based primarily on the DBM cost of the MPPI weighted output, with distribution
quality and stability terms.  Warm is always in the bank, so T1 can fall back
instead of forcing a worse teacher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_proposal_teacher import (
    DEFAULT_SOURCE,
    discover_snapshots,
    repository_state,
    sha256_file,
    write_json,
)


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-multicenter-teacher-t1-v1"
DEFAULT_T0 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t0_20260803_v1"
)
DEFAULT_CONFIG = Path(__file__).with_name("dbm_teacher_t1_config_20260803_v1.json")
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_20260803_v1"
)
REQUIRED_COST_KEYS = {
    "position",
    "yaw",
    "vx",
    "yawrate",
    "acceleration_rate",
    "steering_rate",
}
INITIAL_CENTER_NAMES = ("warm", "t0_best", "t0_soft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t0-labels", type=Path, default=DEFAULT_T0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Process only the first N snapshots for a pilot; zero means all.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("format_version") != 1:
        raise ValueError("T1 config format_version must be 1")
    objective = config.get("objective", {})
    weights = objective.get("cost_weights", {})
    if set(weights) != REQUIRED_COST_KEYS:
        raise ValueError(f"cost_weights keys must be {sorted(REQUIRED_COST_KEYS)}")
    if float(weights["yawrate"]) != 0.0:
        raise ValueError("schema-v2 reference requires zero yawrate cost weight")
    if float(objective.get("temperature", 0.0)) <= 0:
        raise ValueError("temperature must be positive")
    search = config.get("search", {})
    if tuple(search.get("initial_centers", ())) != INITIAL_CENTER_NAMES:
        raise ValueError(f"initial_centers must be {list(INITIAL_CENTER_NAMES)}")
    seeds = search.get("seeds", [])
    scales = search.get("sigma_scales", [])
    samples = int(search.get("stage_samples", 0))
    if not seeds or not scales:
        raise ValueError("search seeds and sigma_scales cannot be empty")
    if samples < 4 or samples % 2:
        raise ValueError("stage_samples must be an even integer >= 4")
    if any(float(scale) <= 0 for scale in scales):
        raise ValueError("search sigma scales must be positive")
    elite_fraction = float(search.get("elite_fraction", 0.0))
    if not 0 < elite_fraction <= 0.5:
        raise ValueError("elite_fraction must be within (0, 0.5]")
    if int(search.get("shortlist_count", 0)) < len(INITIAL_CENTER_NAMES):
        raise ValueError("shortlist_count must retain all initial centers")
    proposal = config.get("proposal_evaluation", {})
    proposal_count = int(proposal.get("num_samples", 0))
    if (
        proposal_count < 4
        or proposal_count % 2
        or not proposal.get("seeds")
        or not proposal.get("audit_seeds")
    ):
        raise ValueError(
            "proposal evaluation requires selection/audit seeds and an even sample count"
        )
    if set(proposal["seeds"]) & set(proposal["audit_seeds"]):
        raise ValueError("proposal selection and audit seeds must be disjoint")
    score_keys = {
        "weighted_output_cost_mean",
        "weighted_output_cost_std",
        "p10_cost_mean",
        "softmin_cost_mean",
        "center_shift_standardized_rms",
        "boundary_fraction",
    }
    if set(config.get("selection_score", {})) != score_keys:
        raise ValueError(f"selection_score keys must be {sorted(score_keys)}")
    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_seeded_noise(
    seed: int, count: int, shape: tuple[int, int], sigma: np.ndarray
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((count, *shape)).astype(np.float32)
    noise *= sigma.reshape(1, 1, 2).astype(np.float32)
    noise[0] = 0.0
    return noise


def antithetic_candidates(
    center: np.ndarray,
    sigma: np.ndarray,
    count: int,
    rng: np.random.Generator,
    second_anchor: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> np.ndarray:
    pair_count = (count - 2) // 2
    direction = rng.standard_normal((pair_count, *center.shape)).astype(np.float32)
    perturbation = direction * sigma.reshape(1, 1, 2).astype(np.float32)
    candidates = np.concatenate(
        (
            center[None],
            second_anchor[None],
            center[None] + perturbation,
            center[None] - perturbation,
        ),
        axis=0,
    )
    return np.clip(candidates, action_min, action_max).astype(np.float32)


@torch.no_grad()
def evaluate_knots(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    knots: np.ndarray,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots_tensor = torch.from_numpy(np.asarray(knots, dtype=np.float32)).to(
        controller.device
    )
    actions = controller._interpolate_knots(knots_tensor)
    trajectories = backend(history, initial_state, current_action, actions).to(
        controller.device
    )
    cost = controller.trajectory_cost(
        trajectories, actions, reference, current_action
    )
    return (
        cost.detach().cpu().numpy(),
        actions.detach().cpu().numpy(),
        trajectories.detach().cpu().numpy(),
    )


@torch.no_grad()
def evaluate_actions(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    actions: np.ndarray,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    action_tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(
        controller.device
    )
    trajectories = backend(
        history, initial_state, current_action, action_tensor
    ).to(controller.device)
    cost = controller.trajectory_cost(
        trajectories, action_tensor, reference, current_action
    )
    return cost.detach().cpu().numpy(), trajectories.detach().cpu().numpy()


def stable_weight(cost: np.ndarray, temperature: float) -> np.ndarray:
    unnormalized = np.exp(-(cost.astype(np.float64) - float(np.min(cost))) / temperature)
    return unnormalized / np.sum(unnormalized)


def softmin(cost: np.ndarray, temperature: float) -> float:
    minimum = float(np.min(cost))
    shifted_mean = float(np.mean(np.exp(-(cost.astype(np.float64) - minimum) / temperature)))
    return minimum - temperature * math.log(shifted_mean)


def add_unique_center(
    names: list[str], centers: list[np.ndarray], name: str, center: np.ndarray
) -> int:
    value = np.asarray(center, dtype=np.float32)
    for index, existing in enumerate(centers):
        if np.allclose(value, existing, rtol=0, atol=1e-7):
            names[index] = f"{names[index]}|{name}"
            return index
    names.append(name)
    centers.append(value.copy())
    return len(centers) - 1


def objective_matches_snapshot(config: dict[str, Any], data: np.lib.npyio.NpzFile) -> None:
    source_weights = json.loads(str(data["cost_weights_json"]))
    source_params = json.loads(str(data["mppi_params_json"]))
    objective = config["objective"]
    if not np.isclose(float(objective["temperature"]), float(source_params["temperature"])):
        raise ValueError("T1 objective temperature differs from source MPPI")
    for name in REQUIRED_COST_KEYS:
        if not np.isclose(
            float(objective["cost_weights"][name]), float(source_weights[name])
        ):
            raise ValueError(f"T1 objective weight {name} differs from source")


def load_t0_centers(
    t0_root: Path, record: dict[str, Any], source_hash: str
) -> tuple[np.ndarray, np.ndarray, str]:
    label_path = (
        t0_root
        / record["episode_id"]
        / f"step_{record['control_step']:06d}.npz"
    )
    with np.load(label_path, allow_pickle=False) as label:
        if str(label["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{label_path}: T0 source hash mismatch")
        config_ids = list(label["config_ids"].astype(str))
        if "collection_default" not in config_ids:
            raise KeyError(f"{label_path}: collection_default label is unavailable")
        index = config_ids.index("collection_default")
        return (
            np.asarray(label["best_candidate_knots"][index], dtype=np.float32),
            np.asarray(label["soft_teacher_center_knots"][index], dtype=np.float32),
            sha256_file(label_path),
        )


def make_controller(
    data: np.lib.npyio.NpzFile, config: dict[str, Any], device: torch.device
) -> tuple[TorchMPPIController, TorchDynamicBicycleRolloutBackend]:
    params = TorchMPPIParams(**json.loads(str(data["mppi_params_json"])))
    weights = TorchMPPICostWeights(**config["objective"]["cost_weights"])
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(str(data["dbm_params_json"])))
    )
    backend.set_initial_lateral_velocity(float(data["initial_lateral_velocity"]))
    controller = TorchMPPIController(
        backend, params=params, cost_weights=weights, device=device
    )
    return controller, backend


def search_center_pool(
    initial_names: list[str],
    initial_centers: list[np.ndarray],
    config: dict[str, Any],
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[list[str], list[np.ndarray], list[dict[str, Any]]]:
    names: list[str] = []
    centers: list[np.ndarray] = []
    for name, center in zip(initial_names, initial_centers):
        add_unique_center(names, centers, name, center)
    diagnostics: list[dict[str, Any]] = []
    search = config["search"]
    elite_count = max(2, int(math.ceil(search["stage_samples"] * search["elite_fraction"])))
    for start_name, start_center in zip(initial_names, initial_centers):
        for seed in search["seeds"]:
            rng = np.random.default_rng(int(seed))
            center = start_center.copy()
            global_best = center.copy()
            global_best_cost = float("inf")
            for stage_index, scale in enumerate(search["sigma_scales"], start=1):
                stage_sigma = sigma * float(scale)
                knots = antithetic_candidates(
                    center,
                    stage_sigma,
                    int(search["stage_samples"]),
                    rng,
                    global_best,
                    action_min,
                    action_max,
                )
                cost, _, _ = evaluate_knots(
                    controller,
                    backend,
                    knots,
                    history,
                    initial_state,
                    current_action,
                    reference,
                )
                best_index = int(np.argmin(cost))
                if float(cost[best_index]) < global_best_cost:
                    global_best_cost = float(cost[best_index])
                    global_best = knots[best_index].copy()
                elite_indices = np.argsort(cost)[:elite_count]
                elite_cost = cost[elite_indices]
                scale_cost = max(
                    float(np.quantile(cost, 0.10) - np.min(cost)),
                    float(search["elite_temperature_floor"]),
                )
                elite_weight = np.exp(
                    -(elite_cost.astype(np.float64) - float(elite_cost.min()))
                    / scale_cost
                )
                elite_weight /= elite_weight.sum()
                center = np.sum(
                    elite_weight[:, None, None] * knots[elite_indices], axis=0
                ).astype(np.float32)
                center = np.clip(center, action_min, action_max).astype(np.float32)
                center_name = f"cem_{start_name}_seed{seed}_stage{stage_index}"
                add_unique_center(names, centers, center_name, center)
                diagnostics.append(
                    {
                        "start_name": start_name,
                        "seed": int(seed),
                        "stage_index": stage_index,
                        "sigma_scale": float(scale),
                        "best_cost": float(cost[best_index]),
                        "p10_cost": float(np.quantile(cost, 0.10)),
                        "median_cost": float(np.median(cost)),
                        "global_best_cost": global_best_cost,
                        "elite_count": elite_count,
                        "elite_cost_scale": scale_cost,
                        "generated_center_name": center_name,
                    }
                )
    return names, centers, diagnostics


def shortlist_centers(
    names: list[str],
    centers: list[np.ndarray],
    direct_cost: np.ndarray,
    required_names: tuple[str, ...],
    count: int,
) -> np.ndarray:
    selected: list[int] = []
    for required in required_names:
        matches = [index for index, name in enumerate(names) if required in name.split("|")]
        if not matches:
            raise AssertionError(f"required center {required} disappeared from pool")
        if matches[0] not in selected:
            selected.append(matches[0])
    for index in np.argsort(direct_cost):
        value = int(index)
        if value not in selected:
            selected.append(value)
        if len(selected) >= min(count, len(centers)):
            break
    return np.asarray(selected, dtype=np.int32)


def proposal_evaluation(
    centers: np.ndarray,
    config: dict[str, Any],
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    warm: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    evaluation_seeds: list[int] | None = None,
) -> dict[str, np.ndarray]:
    proposal = config["proposal_evaluation"]
    seeds = [
        int(seed)
        for seed in (evaluation_seeds if evaluation_seeds is not None else proposal["seeds"])
    ]
    sample_count = int(proposal["num_samples"])
    proposal_sigma = sigma * float(proposal["sigma_scale"])
    center_count = len(centers)
    seed_count = len(seeds)
    candidate_cost = np.empty((center_count, seed_count, sample_count), np.float32)
    candidate_weight = np.empty_like(candidate_cost)
    weighted_actions = np.empty(
        (center_count, seed_count, controller.params.horizon, 2), np.float32
    )
    weighted_trajectories = np.empty(
        (center_count, seed_count, controller.params.horizon, 5), np.float32
    )
    weighted_cost = np.empty((center_count, seed_count), np.float32)
    best_cost = np.empty_like(weighted_cost)
    p10_cost = np.empty_like(weighted_cost)
    median_cost = np.empty_like(weighted_cost)
    softmin_cost = np.empty_like(weighted_cost)
    ess = np.empty_like(weighted_cost)
    clip_fraction = np.empty_like(weighted_cost)
    temperature = float(config["objective"]["temperature"])
    for seed_index, seed in enumerate(seeds):
        noise = stable_seeded_noise(seed, sample_count, centers[0].shape, proposal_sigma)
        for center_index, center in enumerate(centers):
            raw_knots = center[None] + noise
            knots = np.clip(raw_knots, action_min, action_max).astype(np.float32)
            cost, actions, _ = evaluate_knots(
                controller,
                backend,
                knots,
                history,
                initial_state,
                current_action,
                reference,
            )
            weight = stable_weight(cost, temperature)
            weighted_action = np.sum(weight[:, None, None] * actions, axis=0).astype(
                np.float32
            )
            out_cost, out_trajectory = evaluate_actions(
                controller,
                backend,
                weighted_action[None],
                history,
                initial_state,
                current_action,
                reference,
            )
            candidate_cost[center_index, seed_index] = cost
            candidate_weight[center_index, seed_index] = weight.astype(np.float32)
            weighted_actions[center_index, seed_index] = weighted_action
            weighted_trajectories[center_index, seed_index] = out_trajectory[0]
            weighted_cost[center_index, seed_index] = out_cost[0]
            best_cost[center_index, seed_index] = float(np.min(cost))
            p10_cost[center_index, seed_index] = float(np.quantile(cost, 0.10))
            median_cost[center_index, seed_index] = float(np.median(cost))
            softmin_cost[center_index, seed_index] = softmin(cost, temperature)
            ess[center_index, seed_index] = float(1.0 / np.sum(np.square(weight)))
            clip_fraction[center_index, seed_index] = float(
                np.mean(raw_knots != knots)
            )
    shift_rms = np.sqrt(
        np.mean(np.square((centers - warm[None]) / sigma.reshape(1, 1, 2)), axis=(1, 2))
    )
    boundary_fraction = np.mean(
        (centers <= action_min.reshape(1, 1, 2) + 1e-6)
        | (centers >= action_max.reshape(1, 1, 2) - 1e-6),
        axis=(1, 2),
    )
    score_weights = config["selection_score"]
    score_components = {
        "weighted_output_cost_mean": weighted_cost.mean(axis=1),
        "weighted_output_cost_std": weighted_cost.std(axis=1),
        "p10_cost_mean": p10_cost.mean(axis=1),
        "softmin_cost_mean": softmin_cost.mean(axis=1),
        "center_shift_standardized_rms": shift_rms,
        "boundary_fraction": boundary_fraction,
    }
    selection_score = sum(
        float(score_weights[name]) * values
        for name, values in score_components.items()
    )
    return {
        "proposal_candidate_cost": candidate_cost,
        "proposal_candidate_weight": candidate_weight,
        "proposal_weighted_action_sequences": weighted_actions,
        "proposal_weighted_trajectories": weighted_trajectories,
        "proposal_weighted_output_cost": weighted_cost,
        "proposal_best_cost": best_cost,
        "proposal_p10_cost": p10_cost,
        "proposal_median_cost": median_cost,
        "proposal_softmin_cost": softmin_cost,
        "proposal_effective_sample_size": ess,
        "proposal_clip_fraction": clip_fraction,
        "center_shift_standardized_rms": shift_rms.astype(np.float32),
        "center_boundary_fraction": boundary_fraction.astype(np.float32),
        "selection_score": selection_score.astype(np.float32),
        **{
            f"score_component_{name}": values.astype(np.float32)
            for name, values in score_components.items()
        },
    }


def process_snapshot(
    record: dict[str, Any],
    source_root: Path,
    t0_root: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    source_hash = sha256_file(record["path"])
    t0_best, t0_soft, t0_hash = load_t0_centers(t0_root, record, source_hash)
    with np.load(record["path"], allow_pickle=False) as data:
        objective_matches_snapshot(config, data)
        controller, backend = make_controller(data, config, device)
        warm = np.asarray(data["sampling_mean_knots"], dtype=np.float32)
        initial_centers = [warm, t0_best, t0_soft]
        params = json.loads(str(data["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
        action_min = np.asarray(params["action_min"], dtype=np.float32)
        action_max = np.asarray(params["action_max"], dtype=np.float32)
        history = torch.from_numpy(data["history"]).to(device)
        initial_state = torch.from_numpy(data["initial_state"]).to(device).reshape(1, 5)
        current_action = torch.from_numpy(data["current_action"]).to(device).reshape(1, 2)
        reference = controller._prepare_reference(data["reference"])
        pool_names, pool_centers, search_diagnostics = search_center_pool(
            list(INITIAL_CENTER_NAMES),
            initial_centers,
            config,
            controller,
            backend,
            history,
            initial_state,
            current_action,
            reference,
            sigma,
            action_min,
            action_max,
        )
        pool_array = np.asarray(pool_centers, dtype=np.float32)
        pool_direct_cost, _, _ = evaluate_knots(
            controller,
            backend,
            pool_array,
            history,
            initial_state,
            current_action,
            reference,
        )
        shortlist_indices = shortlist_centers(
            pool_names,
            pool_centers,
            pool_direct_cost,
            INITIAL_CENTER_NAMES,
            int(config["search"]["shortlist_count"]),
        )
        shortlist_names = [pool_names[index] for index in shortlist_indices]
        shortlist = pool_array[shortlist_indices]
        shortlist_direct_cost, shortlist_actions, shortlist_trajectories = evaluate_knots(
            controller,
            backend,
            shortlist,
            history,
            initial_state,
            current_action,
            reference,
        )
        proposal = proposal_evaluation(
            shortlist,
            config,
            controller,
            backend,
            history,
            initial_state,
            current_action,
            reference,
            warm,
            sigma,
            action_min,
            action_max,
        )
        teacher_index = int(np.argmin(proposal["selection_score"]))
        warm_matches = [
            index
            for index, name in enumerate(shortlist_names)
            if "warm" in name.split("|")
        ]
        if len(warm_matches) != 1:
            raise AssertionError("shortlist must contain exactly one warm center")
        warm_index = warm_matches[0]
        teacher_center = shortlist[teacher_index]
        teacher_name = shortlist_names[teacher_index]
        audit_centers = np.asarray((warm, teacher_center), dtype=np.float32)
        audit = proposal_evaluation(
            audit_centers,
            config,
            controller,
            backend,
            history,
            initial_state,
            current_action,
            reference,
            warm,
            sigma,
            action_min,
            action_max,
            evaluation_seeds=[
                int(seed) for seed in config["proposal_evaluation"]["audit_seeds"]
            ],
        )
        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
            "source_snapshot_sha256": np.asarray(source_hash),
            "t0_label_sha256": np.asarray(t0_hash),
            "pool_center_names": np.asarray(pool_names),
            "pool_centers": pool_array,
            "pool_direct_cost": pool_direct_cost.astype(np.float32),
            "shortlist_pool_indices": shortlist_indices,
            "shortlist_center_names": np.asarray(shortlist_names),
            "shortlist_centers": shortlist,
            "shortlist_direct_action_sequences": shortlist_actions,
            "shortlist_direct_trajectories": shortlist_trajectories,
            "shortlist_direct_cost": shortlist_direct_cost.astype(np.float32),
            "proposal_evaluation_seeds": np.asarray(
                config["proposal_evaluation"]["seeds"], dtype=np.int64
            ),
            **proposal,
            "warm_shortlist_index": np.asarray(warm_index, dtype=np.int32),
            "teacher_shortlist_index": np.asarray(teacher_index, dtype=np.int32),
            "teacher_center_source": np.asarray(teacher_name),
            "teacher_center_knots": teacher_center.astype(np.float32),
            "teacher_delta_knots": (teacher_center - warm).astype(np.float32),
            "teacher_direct_cost": np.asarray(
                shortlist_direct_cost[teacher_index], dtype=np.float32
            ),
            "teacher_weighted_action_sequences": proposal[
                "proposal_weighted_action_sequences"
            ][teacher_index],
            "teacher_weighted_trajectories": proposal[
                "proposal_weighted_trajectories"
            ][teacher_index],
            "audit_center_names": np.asarray(("warm", "teacher")),
            "audit_centers": audit_centers,
            "audit_evaluation_seeds": np.asarray(
                config["proposal_evaluation"]["audit_seeds"], dtype=np.int64
            ),
            **{f"audit_{name}": value for name, value in audit.items()},
        }
        metadata = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "source_relative_path": record["source_relative_path"],
            "source_snapshot_sha256": source_hash,
            "t0_label_sha256": t0_hash,
            "episode_id": record["episode_id"],
            "control_step": record["control_step"],
            "pool_center_count": len(pool_names),
            "shortlist_center_count": len(shortlist_names),
            "teacher_center_source": teacher_name,
            "teacher_shortlist_index": teacher_index,
            "warm_shortlist_index": warm_index,
            "search_stages": search_diagnostics,
            "label_semantics": (
                "teacher_center_knots minimizes the configured multi-seed proposal "
                "score among the retained center bank; every proposal weighted output "
                "is independently rolled out by the fixed Torch DBM"
            ),
        }
        warm_score = float(proposal["selection_score"][warm_index])
        teacher_score = float(proposal["selection_score"][teacher_index])
        row = {
            "episode_id": record["episode_id"],
            "control_step": record["control_step"],
            "teacher_center_source": teacher_name,
            "teacher_is_warm": teacher_index == warm_index,
            "warm_selection_score": warm_score,
            "teacher_selection_score": teacher_score,
            "selection_score_improvement": warm_score - teacher_score,
            "warm_weighted_output_cost_mean": float(
                proposal["proposal_weighted_output_cost"][warm_index].mean()
            ),
            "teacher_weighted_output_cost_mean": float(
                proposal["proposal_weighted_output_cost"][teacher_index].mean()
            ),
            "warm_p10_cost_mean": float(
                proposal["proposal_p10_cost"][warm_index].mean()
            ),
            "teacher_p10_cost_mean": float(
                proposal["proposal_p10_cost"][teacher_index].mean()
            ),
            "warm_softmin_cost_mean": float(
                proposal["proposal_softmin_cost"][warm_index].mean()
            ),
            "teacher_softmin_cost_mean": float(
                proposal["proposal_softmin_cost"][teacher_index].mean()
            ),
            "teacher_direct_cost": float(shortlist_direct_cost[teacher_index]),
            "teacher_delta_standardized_rms": float(
                proposal["center_shift_standardized_rms"][teacher_index]
            ),
            "teacher_boundary_fraction": float(
                proposal["center_boundary_fraction"][teacher_index]
            ),
            "audit_warm_weighted_output_cost_mean": float(
                audit["proposal_weighted_output_cost"][0].mean()
            ),
            "audit_teacher_weighted_output_cost_mean": float(
                audit["proposal_weighted_output_cost"][1].mean()
            ),
            "audit_warm_p10_cost_mean": float(
                audit["proposal_p10_cost"][0].mean()
            ),
            "audit_teacher_p10_cost_mean": float(
                audit["proposal_p10_cost"][1].mean()
            ),
            "source_snapshot_sha256": source_hash,
            "t0_label_sha256": t0_hash,
        }
        return arrays, metadata, row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    score_gain = values("selection_score_improvement")
    out_gain = values("warm_weighted_output_cost_mean") - values(
        "teacher_weighted_output_cost_mean"
    )
    p10_gain = values("warm_p10_cost_mean") - values("teacher_p10_cost_mean")
    audit_out_gain = values("audit_warm_weighted_output_cost_mean") - values(
        "audit_teacher_weighted_output_cost_mean"
    )
    audit_p10_gain = values("audit_warm_p10_cost_mean") - values(
        "audit_teacher_p10_cost_mean"
    )
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row["teacher_center_source"])
        sources[source] = sources.get(source, 0) + 1
    return {
        "snapshot_count": len(rows),
        "teacher_is_warm_count": int(sum(bool(row["teacher_is_warm"]) for row in rows)),
        "selection_score_improved_count": int(np.sum(score_gain > 1e-6)),
        "selection_score_improvement": {
            "mean": float(score_gain.mean()),
            "median": float(np.median(score_gain)),
            "p10": float(np.quantile(score_gain, 0.10)),
            "p90": float(np.quantile(score_gain, 0.90)),
        },
        "weighted_output_cost_improvement": {
            "improved_count": int(np.sum(out_gain > 0)),
            "mean": float(out_gain.mean()),
            "median": float(np.median(out_gain)),
            "p10": float(np.quantile(out_gain, 0.10)),
            "p90": float(np.quantile(out_gain, 0.90)),
        },
        "p10_cost_improvement": {
            "improved_count": int(np.sum(p10_gain > 0)),
            "mean": float(p10_gain.mean()),
            "median": float(np.median(p10_gain)),
        },
        "heldout_audit_weighted_output_cost_improvement": {
            "improved_count": int(np.sum(audit_out_gain > 0)),
            "mean": float(audit_out_gain.mean()),
            "median": float(np.median(audit_out_gain)),
            "p10": float(np.quantile(audit_out_gain, 0.10)),
            "p90": float(np.quantile(audit_out_gain, 0.90)),
        },
        "heldout_audit_p10_cost_improvement": {
            "improved_count": int(np.sum(audit_p10_gain > 0)),
            "mean": float(audit_p10_gain.mean()),
            "median": float(np.median(audit_p10_gain)),
        },
        "teacher_delta_standardized_rms": {
            "mean": float(values("teacher_delta_standardized_rms").mean()),
            "median": float(np.median(values("teacher_delta_standardized_rms"))),
            "maximum": float(values("teacher_delta_standardized_rms").max()),
        },
        "teacher_source_counts": dict(sorted(sources.items())),
    }


def write_summary(path: Path, source: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    search_rollouts = (
        len(INITIAL_CENTER_NAMES)
        * len(config["search"]["seeds"])
        * len(config["search"]["sigma_scales"])
        * int(config["search"]["stage_samples"])
    )
    probe_rollouts = (
        int(config["search"]["shortlist_count"])
        * len(config["proposal_evaluation"]["seeds"])
        * int(config["proposal_evaluation"]["num_samples"])
    )
    audit_rollouts = (
        2
        * len(config["proposal_evaluation"]["audit_seeds"])
        * int(config["proposal_evaluation"]["num_samples"])
    )
    lines = [
        "# DBM multi-center teacher T1 summary",
        "",
        f"- Source: `{source}`",
        f"- Snapshots: {summary['snapshot_count']}",
        f"- Search rollouts per snapshot: {search_rollouts}",
        f"- Proposal-probe rollouts per snapshot: {probe_rollouts}",
        f"- Disjoint warm/teacher audit rollouts per snapshot: {audit_rollouts}",
        "- Every probe's MPPI weighted output is separately rolled out by Torch DBM.",
        "",
        "## Aggregate result",
        "",
        f"- Teacher falls back to warm: {summary['teacher_is_warm_count']}/{summary['snapshot_count']}",
        "- Selection-score improvement mean/median: "
        f"{summary['selection_score_improvement']['mean']:.6f} / "
        f"{summary['selection_score_improvement']['median']:.6f}",
        "- Weighted-output cost improvement mean/median: "
        f"{summary['weighted_output_cost_improvement']['mean']:.6f} / "
        f"{summary['weighted_output_cost_improvement']['median']:.6f}",
        "- Weighted-output cost improved snapshots: "
        f"{summary['weighted_output_cost_improvement']['improved_count']}/"
        f"{summary['snapshot_count']}",
        "- P10 cost improvement mean/median: "
        f"{summary['p10_cost_improvement']['mean']:.6f} / "
        f"{summary['p10_cost_improvement']['median']:.6f}",
        "- Held-out audit weighted-output improvement mean/median: "
        f"{summary['heldout_audit_weighted_output_cost_improvement']['mean']:.6f} / "
        f"{summary['heldout_audit_weighted_output_cost_improvement']['median']:.6f}",
        "- Held-out audit weighted-output improved snapshots: "
        f"{summary['heldout_audit_weighted_output_cost_improvement']['improved_count']}/"
        f"{summary['snapshot_count']}",
        "",
        "The score includes weighted-output mean/std, P10, soft-min, center shift, and boundary terms. "
        "A non-warm teacher can therefore trade a very small mean-cost increase for better stability/distribution quality.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    t0_root = args.t0_labels.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable T1 sidecar: {output}")
    for path in (source, t0_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.max_snapshots < 0:
        raise ValueError("max-snapshots cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(config_path)
    _, records = discover_snapshots(source)
    if args.max_snapshots:
        records = records[: args.max_snapshots]
    t0_manifest_path = t0_root / "manifest.json"
    t0_manifest = json.loads(t0_manifest_path.read_text())
    splits = t0_manifest["splits"]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    source_index: list[dict[str, Any]] = []
    try:
        shutil.copy2(config_path, staging / "teacher_config.json")
        write_json(staging / "splits.json", {"format_version": 1, **splits})
        for index, record in enumerate(records, start=1):
            snapshot_start = time.perf_counter()
            arrays, metadata, row = process_snapshot(
                record, source, t0_root, config, device
            )
            episode_dir = staging / record["episode_id"]
            episode_dir.mkdir(exist_ok=True)
            stem = f"step_{record['control_step']:06d}"
            label_path = episode_dir / f"{stem}.npz"
            np.savez_compressed(label_path, **arrays)
            write_json(episode_dir / f"{stem}.json", metadata)
            rows.append(row)
            source_index.append(
                {
                    "source_relative_path": record["source_relative_path"],
                    "source_snapshot_sha256": row["source_snapshot_sha256"],
                    "t0_label_sha256": row["t0_label_sha256"],
                    "label_relative_path": str(label_path.relative_to(staging)),
                }
            )
            print(
                f"[{index:03d}/{len(records):03d}] {record['source_relative_path']} "
                f"teacher={row['teacher_center_source']} "
                f"score_gain={row['selection_score_improvement']:.4f} "
                f"seconds={time.perf_counter() - snapshot_start:.2f}"
            )
        with (staging / "labels.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = aggregate(rows)
        summary_payload = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "source_collection": str(source),
            "t0_labels": str(t0_root),
            "elapsed_seconds": time.perf_counter() - started,
            **summary,
        }
        write_json(staging / "summary.json", summary_payload)
        write_summary(staging / "COLLECTION_SUMMARY.md", source, config, summary)
        fingerprint_lines = [
            f"{item['source_relative_path']} {item['source_snapshot_sha256']} {item['t0_label_sha256']}"
            for item in source_index
        ]
        write_json(
            staging / "manifest.json",
            {
                "format_version": FORMAT_VERSION,
                "dataset_type": "anycar-dbm-multicenter-teacher-sidecar",
                "generator_id": GENERATOR_ID,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_collection": str(source),
                "t0_labels": str(t0_root),
                "snapshot_count": len(records),
                "source_fingerprint_sha256": hashlib.sha256(
                    "\n".join(fingerprint_lines).encode()
                ).hexdigest(),
                "source_index": source_index,
                "teacher_config_source": str(config_path),
                "teacher_config_sha256": sha256_file(config_path),
                "embedded_teacher_config_sha256": sha256_file(
                    staging / "teacher_config.json"
                ),
                "generator_sha256": sha256_file(Path(__file__).resolve()),
                "t0_manifest_sha256": sha256_file(t0_manifest_path),
                "repository": repository_state(Path(__file__).resolve().parents[2]),
                "device": str(device),
                "splits": splits,
                "config": config,
                "limitations": [
                    "The center bank is finite and cannot prove global optimality.",
                    "Teacher selection uses three fixed proposal seeds; held-out seeds are still required before BC claims.",
                    "The current source collection covers one track, fixed DBM dynamics, and 96 state contexts.",
                ],
            },
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "output": str(output), **summary_payload}))


if __name__ == "__main__":
    main()
