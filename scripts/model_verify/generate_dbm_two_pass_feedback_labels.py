#!/usr/bin/env python3
"""Generate feedback-conditioned, two-pass DBM MPPI critic labels.

No new simulator states are collected.  At every frozen state, pass one uses
128 antithetic rollouts around the frozen BC center and fits the existing
trajectory-residual Gauss-Newton update.  Pass two evaluates a full-rank bank
around that guided center with narrower MPPI noise.  Selection and audit use
disjoint first/second-pass seed pairs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from compare_guided_mppi_stage_counts import (
    make_device_independent_antithetic_candidates,
)
from generate_dbm_fullrank_local_labels import (
    ACTION_DIMENSION,
    CENTER_COUNT,
    center_names,
    hadamard_directions,
    make_centers,
)
from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots as evaluate_knots_numpy,
    load_config,
    make_controller,
    stable_weight,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from guide_mppi_sampling_from_trajectory_error import (
    RolloutEvaluation,
    evaluate_knots as evaluate_knots_torch,
    fit_guided_center,
)


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-two-pass-feedback-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_fullrank_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
FEEDBACK_NAMES = tuple(
    [f"guided_step_{index:02d}" for index in range(ACTION_DIMENSION)]
    + [f"empirical_cost_gradient_{index:02d}" for index in range(ACTION_DIMENSION)]
    + [f"empirical_hessian_diagonal_{index:02d}" for index in range(ACTION_DIMENSION)]
    + [f"weighted_first_pass_shift_{index:02d}" for index in range(ACTION_DIMENSION)]
    + [
        "first_base_cost",
        "first_best_cost",
        "first_p10_cost",
        "first_median_cost",
        "first_mean_cost",
        "first_weighted_output_cost",
        "first_softmin_cost",
        "first_effective_sample_fraction",
        "first_clip_fraction",
        "relative_weighted_fit_error",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-first-seeds", default="24001,24002")
    parser.add_argument("--selection-second-seeds", default="24101,24102")
    parser.add_argument("--audit-first-seeds", default="24201,24202")
    parser.add_argument("--audit-second-seeds", default="24301,24302")
    parser.add_argument("--first-samples", type=int, default=128)
    parser.add_argument("--second-samples", type=int, default=64)
    parser.add_argument("--second-noise-scale", type=float, default=0.10)
    parser.add_argument("--radius-sigma", type=float, default=0.15)
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--max-standardized-step", type=float, default=1.0)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed lists must contain distinct integers")
    return seeds


def antithetic_noise(
    seed: int, count: int, shape: tuple[int, ...], sigma: np.ndarray
) -> np.ndarray:
    if count < 4 or count % 2:
        raise ValueError("sample count must be an even integer >=4")
    rng = np.random.default_rng(seed)
    extra = rng.standard_normal(shape).astype(np.float32)
    pairs = rng.standard_normal(((count - 2) // 2, *shape)).astype(np.float32)
    unit = np.concatenate(
        (np.zeros((1, *shape), np.float32), extra[None], pairs, -pairs), axis=0
    )
    return unit * sigma.reshape(1, 1, 2).astype(np.float32)


def second_pass_evaluation(
    centers: np.ndarray,
    seed: int,
    sample_count: int,
    local_sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    temperature: float,
    controller: Any,
    backend: Any,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, np.ndarray]:
    noise = antithetic_noise(seed, sample_count, tuple(centers.shape[1:]), local_sigma)
    raw = centers[:, None] + noise[None]
    knots = np.clip(raw, action_min, action_max).astype(np.float32)
    cost, actions, _ = evaluate_knots_numpy(
        controller, backend, knots.reshape(-1, 8, 2), history,
        initial_state, current_action, reference,
    )
    cost = cost.reshape(CENTER_COUNT, sample_count)
    actions = actions.reshape(CENTER_COUNT, sample_count, *actions.shape[1:])
    shifted = cost.astype(np.float64) - cost.min(axis=1, keepdims=True)
    unnormalized = np.exp(-shifted / temperature)
    weight = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    weighted_actions = np.sum(weight[..., None, None] * actions, axis=1).astype(np.float32)
    output_cost, _ = evaluate_actions(
        controller, backend, weighted_actions, history, initial_state,
        current_action, reference,
    )
    return {
        "proposal_weighted_output_cost": output_cost.astype(np.float32),
        "proposal_best_cost": cost.min(axis=1).astype(np.float32),
        "proposal_p10_cost": np.quantile(cost, 0.10, axis=1).astype(np.float32),
        "proposal_median_cost": np.median(cost, axis=1).astype(np.float32),
        "proposal_effective_sample_size": (1.0 / np.sum(weight * weight, axis=1)).astype(np.float32),
        "proposal_clip_fraction": np.mean(raw != knots, axis=(1, 2, 3)).astype(np.float32),
    }


def one_replicate(
    base: np.ndarray,
    first_seed: int,
    second_seed: int,
    args: argparse.Namespace,
    directions: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    temperature: float,
    controller: Any,
    backend: Any,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, np.ndarray]:
    center_t = torch.from_numpy(base).to(controller.device)
    sigma_t = torch.from_numpy(sigma).to(controller.device)
    first_knots = make_device_independent_antithetic_candidates(
        center_t, sigma_t, args.first_samples, np.random.default_rng(first_seed),
        None, tuple(action_min), tuple(action_max),
    )
    first = evaluate_knots_torch(
        controller, backend, first_knots, history, initial_state,
        current_action, reference,
    )
    guided, guided_step, response, fit = fit_guided_center(
        first, center_t, sigma_t, args.fit_ridge, args.step_damping,
        args.max_standardized_step,
    )
    first_cost = first.cost.detach().cpu().numpy().astype(np.float32)
    first_knots_np = first.knots.detach().cpu().numpy().astype(np.float32)
    weight = stable_weight(first_cost, temperature)
    weighted_actions = np.sum(
        weight[:, None, None] * first.actions.detach().cpu().numpy(), axis=0
    ).astype(np.float32)
    weighted_output_cost, _ = evaluate_actions(
        controller, backend, weighted_actions[None], history, initial_state,
        current_action, reference,
    )
    normalized_delta = ((first_knots_np - base) / sigma.reshape(1, 1, 2)).reshape(
        args.first_samples, ACTION_DIMENSION
    )
    weighted_shift = np.sum(weight[:, None] * normalized_delta, axis=0)
    baseline_residual = first.residuals[0]
    empirical_gradient = (2.0 * (response @ baseline_residual)).detach().cpu().numpy()
    empirical_hessian_diagonal = (
        2.0 * torch.sum(response.square(), dim=1)
    ).detach().cpu().numpy()
    raw_first = base[None] + antithetic_noise(
        first_seed, args.first_samples, tuple(base.shape), sigma
    )
    softmin = float(
        first_cost.min()
        - temperature * np.log(np.mean(np.exp(-(first_cost - first_cost.min()) / temperature)))
    )
    scalar = np.asarray(
        [
            first_cost[0], first_cost.min(), np.quantile(first_cost, 0.10),
            np.median(first_cost), first_cost.mean(), weighted_output_cost[0],
            softmin, 1.0 / np.sum(weight * weight) / args.first_samples,
            np.mean(raw_first != first_knots_np), fit["relative_weighted_fit_error"],
        ], dtype=np.float32,
    )
    feedback = np.concatenate(
        (
            guided_step.detach().cpu().numpy().reshape(-1),
            empirical_gradient.reshape(-1), empirical_hessian_diagonal.reshape(-1),
            weighted_shift.reshape(-1), scalar,
        )
    ).astype(np.float32)
    guided_np = guided.detach().cpu().numpy().astype(np.float32)
    raw_centers, centers = make_centers(
        guided_np, directions, sigma, args.radius_sigma, action_min, action_max
    )
    second = second_pass_evaluation(
        centers, second_seed, args.second_samples,
        sigma * args.second_noise_scale, action_min, action_max, temperature,
        controller, backend, history, initial_state, current_action, reference,
    )
    return {
        "first_pass_seed": np.asarray(first_seed, np.int64),
        "second_pass_seed": np.asarray(second_seed, np.int64),
        "first_pass_knots": first_knots_np,
        "first_pass_cost": first_cost,
        "first_pass_feedback": feedback,
        "guided_center_knots": guided_np,
        "raw_centers": raw_centers,
        "centers": centers,
        **second,
    }


def stack_replicates(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([value[name] for value in values]) for name in values[0]}


def batched_replicates(
    base: np.ndarray,
    seed_pairs: list[tuple[int, int]],
    args: argparse.Namespace,
    directions: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    temperature: float,
    controller: Any,
    backend: Any,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Evaluate all repeats of one partition in two large rollout batches."""
    center_t = torch.from_numpy(base).to(controller.device)
    sigma_t = torch.from_numpy(sigma).to(controller.device)
    first_knots_by_repeat = [
        make_device_independent_antithetic_candidates(
            center_t, sigma_t, args.first_samples, np.random.default_rng(first_seed),
            None, tuple(action_min), tuple(action_max),
        )
        for first_seed, _ in seed_pairs
    ]
    first_all = evaluate_knots_torch(
        controller, backend, torch.cat(first_knots_by_repeat), history,
        initial_state, current_action, reference,
    )
    preliminary: list[dict[str, np.ndarray]] = []
    center_banks = []
    for repeat_index, (first_seed, second_seed) in enumerate(seed_pairs):
        start = repeat_index * args.first_samples
        stop = start + args.first_samples
        first = RolloutEvaluation(
            knots=first_all.knots[start:stop],
            actions=first_all.actions[start:stop],
            trajectories=first_all.trajectories[start:stop],
            cost=first_all.cost[start:stop],
            components={name: value[start:stop] for name, value in first_all.components.items()},
            residuals=first_all.residuals[start:stop],
        )
        guided, guided_step, response, fit = fit_guided_center(
            first, center_t, sigma_t, args.fit_ridge, args.step_damping,
            args.max_standardized_step,
        )
        first_cost = first.cost.detach().cpu().numpy().astype(np.float32)
        first_knots_np = first.knots.detach().cpu().numpy().astype(np.float32)
        weight = stable_weight(first_cost, temperature)
        weighted_actions = np.sum(
            weight[:, None, None] * first.actions.detach().cpu().numpy(), axis=0
        ).astype(np.float32)
        weighted_output_cost, _ = evaluate_actions(
            controller, backend, weighted_actions[None], history, initial_state,
            current_action, reference,
        )
        normalized_delta = ((first_knots_np - base) / sigma.reshape(1, 1, 2)).reshape(
            args.first_samples, ACTION_DIMENSION
        )
        weighted_shift = np.sum(weight[:, None] * normalized_delta, axis=0)
        baseline_residual = first.residuals[0]
        empirical_gradient = (2.0 * (response @ baseline_residual)).detach().cpu().numpy()
        empirical_hessian_diagonal = (
            2.0 * torch.sum(response.square(), dim=1)
        ).detach().cpu().numpy()
        raw_first = base[None] + antithetic_noise(
            first_seed, args.first_samples, tuple(base.shape), sigma
        )
        softmin = float(
            first_cost.min()
            - temperature * np.log(np.mean(np.exp(-(first_cost - first_cost.min()) / temperature)))
        )
        scalar = np.asarray(
            [
                first_cost[0], first_cost.min(), np.quantile(first_cost, 0.10),
                np.median(first_cost), first_cost.mean(), weighted_output_cost[0],
                softmin, 1.0 / np.sum(weight * weight) / args.first_samples,
                np.mean(raw_first != first_knots_np), fit["relative_weighted_fit_error"],
            ], dtype=np.float32,
        )
        feedback = np.concatenate(
            (
                guided_step.detach().cpu().numpy().reshape(-1),
                empirical_gradient.reshape(-1), empirical_hessian_diagonal.reshape(-1),
                weighted_shift.reshape(-1), scalar,
            )
        ).astype(np.float32)
        guided_np = guided.detach().cpu().numpy().astype(np.float32)
        raw_centers, centers = make_centers(
            guided_np, directions, sigma, args.radius_sigma, action_min, action_max
        )
        center_banks.append(centers)
        preliminary.append(
            {
                "first_pass_seed": np.asarray(first_seed, np.int64),
                "second_pass_seed": np.asarray(second_seed, np.int64),
                "first_pass_knots": first_knots_np,
                "first_pass_cost": first_cost,
                "first_pass_feedback": feedback,
                "guided_center_knots": guided_np,
                "raw_centers": raw_centers,
                "centers": centers,
            }
        )

    centers = np.asarray(center_banks, dtype=np.float32)
    local_sigma = sigma * args.second_noise_scale
    noise = np.asarray(
        [
            antithetic_noise(second_seed, args.second_samples, (8, 2), local_sigma)
            for _, second_seed in seed_pairs
        ], dtype=np.float32,
    )
    raw = centers[:, :, None] + noise[:, None]
    knots = np.clip(raw, action_min, action_max).astype(np.float32)
    cost, actions, _ = evaluate_knots_numpy(
        controller, backend, knots.reshape(-1, 8, 2), history,
        initial_state, current_action, reference,
    )
    repeat_count = len(seed_pairs)
    cost = cost.reshape(repeat_count, CENTER_COUNT, args.second_samples)
    actions = actions.reshape(
        repeat_count, CENTER_COUNT, args.second_samples, *actions.shape[1:]
    )
    shifted = cost.astype(np.float64) - cost.min(axis=2, keepdims=True)
    unnormalized = np.exp(-shifted / temperature)
    weight = unnormalized / unnormalized.sum(axis=2, keepdims=True)
    weighted_actions = np.sum(weight[..., None, None] * actions, axis=2).astype(np.float32)
    output_cost, _ = evaluate_actions(
        controller, backend, weighted_actions.reshape(-1, *weighted_actions.shape[2:]),
        history, initial_state, current_action, reference,
    )
    output_cost = output_cost.reshape(repeat_count, CENTER_COUNT)
    for index, result in enumerate(preliminary):
        result.update(
            {
                "proposal_weighted_output_cost": output_cost[index].astype(np.float32),
                "proposal_best_cost": cost[index].min(axis=1).astype(np.float32),
                "proposal_p10_cost": np.quantile(cost[index], 0.10, axis=1).astype(np.float32),
                "proposal_median_cost": np.median(cost[index], axis=1).astype(np.float32),
                "proposal_effective_sample_size": (
                    1.0 / np.sum(weight[index] * weight[index], axis=1)
                ).astype(np.float32),
                "proposal_clip_fraction": np.mean(
                    raw[index] != knots[index], axis=(1, 2, 3)
                ).astype(np.float32),
            }
        )
    return stack_replicates(preliminary)


def process_one(
    parent_path: Path, source_root: Path, config: dict[str, Any],
    args: argparse.Namespace, directions: np.ndarray,
    selection_pairs: list[tuple[int, int]], audit_pairs: list[tuple[int, int]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episode_id = parent_path.parent.name
    source_path = source_root / episode_id / "snapshots" / parent_path.name
    source_hash, parent_hash = sha256_file(source_path), sha256_file(parent_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(parent_path, allow_pickle=False) as parent:
        if str(parent["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{parent_path}: source hash mismatch")
        base = np.asarray(parent["base_center_knots"], np.float32)
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], np.float32)
        action_min = np.asarray(params["action_min"], np.float32)
        action_max = np.asarray(params["action_max"], np.float32)
        controller, backend = make_controller(source, config, device)
        common = dict(
            base=base, args=args, directions=directions, sigma=sigma,
            action_min=action_min, action_max=action_max,
            temperature=float(config["objective"]["temperature"]),
            controller=controller, backend=backend,
            history=torch.from_numpy(source["history"]).to(device),
            initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
            current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
            reference=controller._prepare_reference(source["reference"]),
        )
        selection = batched_replicates(seed_pairs=selection_pairs, **common)
        audit = batched_replicates(seed_pairs=audit_pairs, **common)
    arrays = {
        "format_version": np.asarray(FORMAT_VERSION, np.int32),
        "source_snapshot_sha256": np.asarray(source_hash),
        "parent_label_sha256": np.asarray(parent_hash),
        "base_center_knots": base,
        "normalized_directions": directions,
        "center_names": np.asarray(center_names()),
        "feedback_names": np.asarray(FEEDBACK_NAMES),
        **selection,
        **{f"audit_{name}": value for name, value in audit.items()},
    }
    metadata = {
        "format_version": FORMAT_VERSION, "generator_id": GENERATOR_ID,
        "episode_id": episode_id,
        "control_step": int(parent_path.stem.removeprefix("step_")),
        "source_snapshot_sha256": source_hash, "parent_label_sha256": parent_hash,
        "selection_seed_pairs": selection_pairs, "audit_seed_pairs": audit_pairs,
    }
    return arrays, metadata


def main() -> None:
    args = parse_args()
    seed_groups = [
        parse_seeds(args.selection_first_seeds), parse_seeds(args.selection_second_seeds),
        parse_seeds(args.audit_first_seeds), parse_seeds(args.audit_second_seeds),
    ]
    if len({len(group) for group in seed_groups}) != 1:
        raise ValueError("all seed lists must have equal length")
    if len(set().union(*map(set, seed_groups))) != sum(map(len, seed_groups)):
        raise ValueError("first/second selection/audit seeds must be disjoint")
    selection_pairs = list(zip(seed_groups[0], seed_groups[1]))
    audit_pairs = list(zip(seed_groups[2], seed_groups[3]))
    output, source_root, parent_root = args.output.resolve(), args.source.resolve(), args.parent_labels.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(parent_root / "teacher_config.json")
    paths = sorted(parent_root.glob("episode_*/*.npz"))
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise FileNotFoundError(parent_root)
    directions = hadamard_directions()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    guided_cost, first_output, ranks = [], [], []
    try:
        for index, parent_path in enumerate(paths, 1):
            arrays, metadata = process_one(
                parent_path, source_root, config, args, directions,
                selection_pairs, audit_pairs, torch.device(args.device),
            )
            episode_dir = staging / parent_path.parent.name
            episode_dir.mkdir(exist_ok=True)
            target = episode_dir / parent_path.name
            np.savez_compressed(target, **arrays)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            guided_cost.append(arrays["proposal_weighted_output_cost"][:, 0])
            first_output.append(arrays["first_pass_feedback"][:, -5])
            for centers, guided in zip(arrays["centers"], arrays["guided_center_knots"]):
                design = (centers - guided).reshape(CENTER_COUNT, -1)
                ranks.append(np.linalg.matrix_rank(design))
            if index % 20 == 0 or index == len(paths):
                print(f"[{index:04d}/{len(paths):04d}] two-pass labels ({time.perf_counter()-started:.1f}s)", flush=True)
        shutil.copy2(parent_root / "splits.json", staging / "splits.json")
        shutil.copy2(parent_root / "teacher_config.json", staging / "teacher_config.json")
        first_array, guided_array = np.asarray(first_output), np.asarray(guided_cost)
        summary = {
            "format_version": FORMAT_VERSION, "generator_id": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root), "parent_labels": str(parent_root),
            "snapshot_count": len(paths), "replicates_per_partition": len(selection_pairs),
            "selection_seed_pairs": selection_pairs, "audit_seed_pairs": audit_pairs,
            "first_samples": args.first_samples, "second_samples_per_center": args.second_samples,
            "second_noise_scale": args.second_noise_scale, "radius_sigma": args.radius_sigma,
            "fit_ridge": args.fit_ridge, "step_damping": args.step_damping,
            "max_standardized_step": args.max_standardized_step,
            "feedback_dimension": len(FEEDBACK_NAMES), "feedback_names": FEEDBACK_NAMES,
            "local_rank_minimum": int(np.min(ranks)),
            "first_weighted_output_cost_mean": float(first_array.mean()),
            "guided_second_weighted_output_cost_mean": float(guided_array.mean()),
            "guided_minus_first_cost_mean": float((guided_array-first_array).mean()),
            "guided_improvement_fraction": float(np.mean(guided_array < first_array)),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "semantics": "Fixed states; first-pass rollout feedback conditions full-rank second-pass center labels. No analytic DBM gradients are used.",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
