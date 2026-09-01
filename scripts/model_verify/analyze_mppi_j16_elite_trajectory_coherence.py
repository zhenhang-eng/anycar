#!/usr/bin/env python3
"""Compare action- and rollout-space coherence of per-state J16 elites.

This is a train-only diagnostic.  It replays the three lowest-cost optimized
knot solutions saved for each physical snapshot and measures whether distant
action-space basins produce a substantially more canonical trajectory effect.
Formal validation and test artifacts are never loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights


DEFAULT_GT = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/j16_elite_trajectory_coherence_20260820_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(count, k=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def wrapped(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def variance_ratio(features: np.ndarray) -> dict[str, float]:
    """Return elite-within variance relative to between-state mean variance."""
    within = float(np.mean(np.var(features, axis=1, ddof=0)))
    between = float(np.mean(np.var(np.mean(features, axis=1), axis=0, ddof=0)))
    return {
        "within_variance": within,
        "between_state_mean_variance": between,
        "within_over_between": within / max(between, 1e-12),
    }


def subgroup_summary(mask: np.ndarray, action_spread: np.ndarray,
                     trajectory_relative: np.ndarray,
                     position_spread: np.ndarray,
                     terminal_position: np.ndarray,
                     terminal_yaw: np.ndarray,
                     episodes: np.ndarray,
                     bootstrap: int,
                     rng: np.random.Generator) -> dict:
    unique_episodes = np.unique(episodes)
    boot_trajectory = []
    boot_position = []
    for _ in range(bootstrap):
        sampled = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        indices = np.concatenate([
            np.flatnonzero((episodes == episode) & mask) for episode in sampled
        ])
        if len(indices) == 0:
            continue
        boot_trajectory.append(float(np.median(trajectory_relative[indices])))
        boot_position.append(float(np.median(position_spread[indices])))
    return {
        "count": int(np.sum(mask)),
        "action_pair_l2_sigma": summarize(action_spread[mask]),
        "trajectory_pair_relative_to_tracking": summarize(
            trajectory_relative[mask]
        ),
        "position_pair_rms_m": summarize(position_spread[mask]),
        "terminal_position_pair_m": summarize(terminal_position[mask]),
        "terminal_yaw_pair_rad": summarize(terminal_yaw[mask]),
        "episode_bootstrap_ci95": {
            "trajectory_pair_relative_median": [
                float(np.quantile(boot_trajectory, 0.025)),
                float(np.quantile(boot_trajectory, 0.975)),
            ],
            "position_pair_rms_m_median": [
                float(np.quantile(boot_position, 0.025)),
                float(np.quantile(boot_position, 0.975)),
            ],
        },
    }


def main() -> None:
    args = parse_args()
    if args.elite_count < 2:
        raise ValueError("elite-count must be at least two")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary_path = args.gt_dir / "summary.json"
    validation_path = args.gt_dir / "validation.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("split") != "train":
        raise ValueError("diagnostic is restricted to the train GT artifact")
    rows = summary["rows"]
    if len(rows) != int(summary["snapshot_count"]):
        raise AssertionError("summary row count mismatch")

    device = torch.device(args.device)
    all_knots: list[np.ndarray] = []
    all_costs: list[np.ndarray] = []
    all_initial: list[np.ndarray] = []
    all_current: list[np.ndarray] = []
    all_reference: list[np.ndarray] = []
    state_keys: list[str] = []
    source_hashes: list[str] = []
    dbm_params_json = None
    cost_weights_json = None
    noise_sigma = None
    for row in rows:
        result_path = Path(row["result"])
        with np.load(result_path, allow_pickle=False) as result:
            order = np.argsort(result["knot_cost_replay"])[:args.elite_count]
            all_knots.append(np.asarray(result["optimized_knots"][order], np.float32))
            all_costs.append(np.asarray(result["knot_cost_replay"][order], np.float32))
            source_path = Path(str(result["source_snapshot"]))
            expected_hash = str(result["source_sha256"])
        with np.load(source_path, allow_pickle=False) as source:
            if sha256(source_path) != expected_hash:
                raise AssertionError(f"source hash mismatch: {source_path}")
            initial = np.asarray(source["initial_state_six"], np.float32)
            current = np.asarray(source["current_action"], np.float32)
            reference = np.asarray(source["reference"], np.float32)
            mppi_params = json.loads(str(source["mppi_params_json"]))
            if len(reference) == int(mppi_params["horizon"]) + 1:
                reference = reference[1:]
            current_dbm = str(source["dbm_params_json"])
            current_weights = str(source["cost_weights_json"])
            current_sigma = np.asarray(mppi_params["noise_sigma"], np.float32)
        if dbm_params_json is None:
            dbm_params_json = current_dbm
            cost_weights_json = current_weights
            noise_sigma = current_sigma
        elif (
            current_dbm != dbm_params_json
            or current_weights != cost_weights_json
            or not np.array_equal(current_sigma, noise_sigma)
        ):
            raise ValueError("DBM, cost, or noise scale changed across snapshots")
        all_initial.append(initial)
        all_current.append(current)
        all_reference.append(reference)
        state_keys.append(f"{row['episode']}#{row['snapshot']}")
        source_hashes.append(expected_hash)

    knots = np.stack(all_knots)
    costs = np.stack(all_costs)
    initial = np.stack(all_initial)
    current = np.stack(all_current)
    reference = np.stack(all_reference)
    assert dbm_params_json is not None and cost_weights_json is not None
    assert noise_sigma is not None
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(dbm_params_json))
    )
    weights = TorchMPPICostWeights(**json.loads(cost_weights_json))
    horizon = int(reference.shape[1])
    trajectories = []
    actions_all = []
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        knot_tensor = torch.as_tensor(knots[start:stop], device=device)
        actions = torch.nn.functional.interpolate(
            knot_tensor.reshape(-1, knots.shape[2], 2).transpose(1, 2),
            size=horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).reshape(stop - start, args.elite_count, horizon, 2)
        initial_tensor = torch.as_tensor(initial[start:stop], device=device)
        flat_initial = initial_tensor[:, None].expand(
            -1, args.elite_count, -1
        ).reshape(-1, 6)
        with torch.no_grad():
            full = backend.rollout_full_state_differentiable(
                flat_initial, actions.reshape(-1, horizon, 2)
            ).reshape(stop - start, args.elite_count, horizon, 6)
        trajectories.append(full.cpu().numpy())
        actions_all.append(actions.cpu().numpy())
    trajectory = np.concatenate(trajectories)
    actions = np.concatenate(actions_all)

    ref = reference[:, None]
    ref_yaw = ref[..., 2]
    dx = trajectory[..., 0] - ref[..., 0]
    dy = trajectory[..., 1] - ref[..., 1]
    cos_yaw = np.cos(ref_yaw)
    sin_yaw = np.sin(ref_yaw)
    along = cos_yaw * dx + sin_yaw * dy
    cross = -sin_yaw * dx + cos_yaw * dy
    yaw_error = np.arctan2(
        np.sin(trajectory[..., 2] - ref_yaw),
        np.cos(trajectory[..., 2] - ref_yaw),
    )
    vx_error = trajectory[..., 3] - ref[..., 3]
    if reference.shape[-1] >= 5:
        yawrate_error = trajectory[..., 5] - ref[..., 4]
    else:
        yawrate_error = trajectory[..., 5]
    effect = np.stack((along, cross, yaw_error, vx_error, yawrate_error), axis=-1)

    left, right = pair_indices(args.elite_count)
    action_delta = (knots[:, left] - knots[:, right]) / noise_sigma
    action_pair_rms = np.sqrt(np.mean(np.square(action_delta), axis=(2, 3)))
    action_pair_l2 = np.sqrt(np.sum(np.square(action_delta), axis=(2, 3)))
    effect_delta = effect[:, left] - effect[:, right]
    position_pair_rms = np.sqrt(np.mean(
        np.square(effect_delta[..., 0]) + np.square(effect_delta[..., 1]), axis=2
    ))
    along_pair_rms = np.sqrt(np.mean(np.square(effect_delta[..., 0]), axis=2))
    cross_pair_rms = np.sqrt(np.mean(np.square(effect_delta[..., 1]), axis=2))
    yaw_pair_rms = np.sqrt(np.mean(np.square(effect_delta[..., 2]), axis=2))
    vx_pair_rms = np.sqrt(np.mean(np.square(effect_delta[..., 3]), axis=2))
    yawrate_pair_rms = np.sqrt(np.mean(np.square(effect_delta[..., 4]), axis=2))
    terminal_position_pair = np.sqrt(
        np.square(effect_delta[..., -1, 0]) + np.square(effect_delta[..., -1, 1])
    )
    terminal_yaw_pair = np.abs(effect_delta[..., -1, 2])

    trajectory_pair_metric = np.sqrt(
        weights.position * np.sum(
            np.square(effect_delta[..., 0]) + np.square(effect_delta[..., 1]), axis=2
        )
        + weights.yaw * np.sum(np.square(effect_delta[..., 2]), axis=2)
        + weights.vx * np.sum(np.square(effect_delta[..., 3]), axis=2)
        + weights.yawrate * np.sum(np.square(effect_delta[..., 4]), axis=2)
    )
    tracking_metric = np.sqrt(
        weights.position * np.sum(
            np.square(effect[..., 0]) + np.square(effect[..., 1]), axis=2
        )
        + weights.yaw * np.sum(np.square(effect[..., 2]), axis=2)
        + weights.vx * np.sum(np.square(effect[..., 3]), axis=2)
        + weights.yawrate * np.sum(np.square(effect[..., 4]), axis=2)
    )
    pair_tracking_scale = 0.5 * (
        tracking_metric[:, left] + tracking_metric[:, right]
    )
    trajectory_pair_relative = trajectory_pair_metric / np.maximum(
        pair_tracking_scale, 1e-6
    )

    # Cost-vector decomposition checks whether equal totals hide different tradeoffs.
    previous = np.concatenate((
        np.broadcast_to(current[:, None, None], (len(rows), args.elite_count, 1, 2)),
        actions[:, :, :-1],
    ), axis=2)
    rate = actions - previous
    cost_terms = np.stack((
        weights.position * np.sum(np.square(effect[..., 0]) + np.square(effect[..., 1]), axis=2),
        weights.yaw * np.sum(np.square(effect[..., 2]), axis=2),
        weights.vx * np.sum(np.square(effect[..., 3]), axis=2),
        weights.yawrate * np.sum(np.square(effect[..., 4]), axis=2),
        weights.acceleration_rate * np.sum(np.square(rate[..., 0]), axis=2),
        weights.steering_rate * np.sum(np.square(rate[..., 1]), axis=2),
    ), axis=-1)
    replay_cost = np.sum(cost_terms, axis=-1)
    replay_error = np.abs(replay_cost - costs)
    if float(np.max(replay_error)) > 1e-3:
        raise AssertionError(f"cost replay mismatch: {np.max(replay_error)}")
    cost_term_pair_l1 = np.sum(
        np.abs(cost_terms[:, left] - cost_terms[:, right]), axis=-1
    )
    total_cost_pair_abs = np.abs(costs[:, left] - costs[:, right])

    action_flat = knots.reshape(len(rows), args.elite_count, -1) / np.tile(
        noise_sigma, knots.shape[2]
    )
    effect_flat = effect.reshape(len(rows), args.elite_count, -1)
    action_variance = variance_ratio(action_flat)
    effect_variance = variance_ratio(effect_flat)

    per_state = {
        "state_keys": np.asarray(state_keys),
        "elite_cost": costs.astype(np.float32),
        "action_pair_rms_sigma": action_pair_rms.astype(np.float32),
        "action_pair_l2_sigma": action_pair_l2.astype(np.float32),
        "position_pair_rms_m": position_pair_rms.astype(np.float32),
        "along_pair_rms_m": along_pair_rms.astype(np.float32),
        "cross_pair_rms_m": cross_pair_rms.astype(np.float32),
        "yaw_pair_rms_rad": yaw_pair_rms.astype(np.float32),
        "vx_pair_rms_mps": vx_pair_rms.astype(np.float32),
        "yawrate_pair_rms_radps": yawrate_pair_rms.astype(np.float32),
        "terminal_position_pair_m": terminal_position_pair.astype(np.float32),
        "terminal_yaw_pair_rad": terminal_yaw_pair.astype(np.float32),
        "trajectory_pair_task_metric": trajectory_pair_metric.astype(np.float32),
        "trajectory_pair_relative_to_tracking": trajectory_pair_relative.astype(np.float32),
        "cost_term_pair_l1": cost_term_pair_l1.astype(np.float32),
        "total_cost_pair_abs": total_cost_pair_abs.astype(np.float32),
        "cost_terms": cost_terms.astype(np.float32),
    }

    # Each summary is over the per-state maximum pair spread: a conservative test.
    spread = {
        key: summarize(np.max(value, axis=1))
        for key, value in per_state.items()
        if key not in ("state_keys", "elite_cost", "cost_terms")
    }
    relative_cost_spread = (
        np.max(costs, axis=1) - np.min(costs, axis=1)
    ) / np.maximum(np.min(costs, axis=1), 1e-6)
    max_action_spread = np.max(action_pair_l2, axis=1)
    max_trajectory_relative = np.max(trajectory_pair_relative, axis=1)
    max_position_spread = np.max(position_pair_rms, axis=1)
    max_terminal_position = np.max(terminal_position_pair, axis=1)
    max_terminal_yaw = np.max(terminal_yaw_pair, axis=1)
    multibasin_masks = {
        "cost_le_0p5pct_action_l2_ge_0p296sigma": (
            (relative_cost_spread <= 0.005) & (max_action_spread >= 0.296)
        ),
        "cost_le_0p5pct_action_l2_ge_0p5sigma": (
            (relative_cost_spread <= 0.005) & (max_action_spread >= 0.5)
        ),
        "cost_le_1pct_action_l2_ge_0p5sigma": (
            (relative_cost_spread <= 0.01) & (max_action_spread >= 0.5)
        ),
    }
    episodes = np.asarray([row["episode"] for row in rows])
    rng = np.random.default_rng(260820)
    multibasin = {
        name: subgroup_summary(
            mask,
            max_action_spread,
            max_trajectory_relative,
            max_position_spread,
            max_terminal_position,
            max_terminal_yaw,
            episodes,
            args.bootstrap,
            rng,
        )
        for name, mask in multibasin_masks.items()
    }
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "J16_ELITE_TRAJECTORY_COHERENCE_DIAGNOSTIC",
        "contract": {
            "split": "train",
            "snapshot_count": len(rows),
            "elite_definition": f"lowest replay-cost top-{args.elite_count} per state",
            "trajectory_effect": "reference-relative along/cross/yaw/vx/yawrate over 50 steps",
            "pair_summary": "maximum pair spread per state, then aggregate",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {
            "gt_summary": str(summary_path.resolve()),
            "gt_summary_sha256": sha256(summary_path),
            "gt_validation": str(validation_path.resolve()),
            "gt_validation_sha256": sha256(validation_path),
            "source_snapshot_sha256_chain": hashlib.sha256(
                "\n".join(source_hashes).encode()
            ).hexdigest(),
        },
        "parameters": {
            "noise_sigma": noise_sigma.tolist(),
            "cost_weights": json.loads(cost_weights_json),
            "device": str(device),
        },
        "checks": {
            "maximum_cost_replay_absolute_error": float(np.max(replay_error)),
            "state_count_matches_summary": len(rows) == int(summary["snapshot_count"]),
        },
        "elite_cost_relative_spread": summarize(relative_cost_spread),
        "maximum_pair_spread_per_state": spread,
        "equal_cost_distinct_action_basin_subgroups": multibasin,
        "within_vs_between_state_variance": {
            "standardized_action_knots": action_variance,
            "reference_relative_trajectory_effect": effect_variance,
            "trajectory_over_action_ratio_of_ratios": (
                effect_variance["within_over_between"]
                / max(action_variance["within_over_between"], 1e-12)
            ),
        },
    }
    action_ratio = action_variance["within_over_between"]
    trajectory_ratio = effect_variance["within_over_between"]
    relative_spread_median = spread["trajectory_pair_relative_to_tracking"]["median"]
    core_multibasin = multibasin["cost_le_0p5pct_action_l2_ge_0p296sigma"]
    multibasin_pass = (
        core_multibasin["count"] >= 100
        and core_multibasin["trajectory_pair_relative_to_tracking"]["median"] <= 0.10
    )
    if (
        trajectory_ratio <= 0.5 * action_ratio
        and relative_spread_median <= 0.25
        and multibasin_pass
    ):
        decision = "TRAJECTORY_EFFECT_SUBSTANTIALLY_MORE_CANONICAL"
        explanation = (
            "Elite actions occupy distinct basins while their reference-relative "
            "rollout effects are substantially more concentrated. A trajectory-"
            "equivalent training objective is justified for a controlled A/B."
        )
    else:
        decision = "TRAJECTORY_EFFECT_NOT_CANONICAL_ENOUGH"
        explanation = (
            "Near-equal-cost action basins also produce materially different "
            "rollout effects. Action-space label ambiguity cannot be removed by "
            "trajectory supervision alone under the current contract."
        )
    result["decision"] = decision
    result["decision_rule"] = {
        "trajectory_within_between_at_most_half_action": trajectory_ratio <= 0.5 * action_ratio,
        "median_max_pair_task_distance_relative_to_tracking_at_most_0_25": relative_spread_median <= 0.25,
        "equal_cost_distinct_basin_subset_count_at_least_100_and_median_relative_distance_at_most_0_10": multibasin_pass,
        "explanation": explanation,
    }

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "per_state.npz", **per_state)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
