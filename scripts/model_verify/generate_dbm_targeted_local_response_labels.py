#!/usr/bin/env python3
"""Generate same-state multidirectional local-response labels on routed DBM contexts.

Every selected context receives 19 outer action-location directions at two
one-sided radii.  At every outer location, a 0.01-sigma full-16D antithetic
finite-difference bank estimates the local transformed-reward gradient.  Only
deterministic forward DBM costs are used; the Actor is frozen and no analytic
DBM gradient is consumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor

from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_local_gradient_critic import fit_local_parameters
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    hadamard_directions,
    make_base_policy,
    residual_outputs,
)
from train_mppi_direct_response_slope_ac import centers_to_actions, transformed
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_LOCAL_LABELS = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2/"
    "local_forward_labels.npz"
)
DEFAULT_ROUTING = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1"
)
STEERING_FLAT_INDICES = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument("--routing-manifest", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-radii-sigma", default="0.05,0.075")
    parser.add_argument("--fd-radius-sigma", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--context-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def response_directions() -> tuple[np.ndarray, list[str]]:
    hadamard = hadamard_directions().reshape(16, 16)
    axes = np.zeros((len(STEERING_FLAT_INDICES), 16), np.float32)
    for row, index in enumerate(STEERING_FLAT_INDICES):
        axes[row, index] = 4.0  # unit RMS in a 16-dimensional coordinate
    directions = np.concatenate((hadamard, axes), axis=0)
    rms = np.sqrt(np.mean(np.square(directions), axis=1, keepdims=True))
    directions = directions / rms
    names = [f"hadamard_{index:02d}" for index in range(16)] + [
        f"front_steering_knot_{index}" for index in STEERING_FLAT_INDICES
    ]
    if np.linalg.matrix_rank(directions) != 16:
        raise AssertionError("response direction bank is not full rank")
    return directions.reshape(-1, 8, 2).astype(np.float32), names


def bounded_action_limits(
    alpha_center: np.ndarray,
    sigma: np.ndarray,
    maximum_residual_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(
        -1.0, alpha_center - maximum_residual_sigma * sigma[:, None, :]
    )
    high = np.minimum(
        1.0, alpha_center + maximum_residual_sigma * sigma[:, None, :]
    )
    return low.astype(np.float32), high.astype(np.float32)


def outer_locations(
    actor_center: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    directions: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = [actor_center[:, None]]
    names = ["actor_center"]
    for radius in radii:
        offset = float(radius) * sigma[:, None, None, :] * directions[None]
        for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
            raw.append(actor_center[:, None] + sign * offset)
            names.extend(
                f"r{float(radius):.3f}_{suffix}_{index:02d}"
                for index in range(len(directions))
            )
    raw_array = np.concatenate(raw, axis=1).astype(np.float32)
    if raw_array.shape[1] != len(names):
        raise AssertionError("outer location/name count mismatch")
    clipped = np.clip(raw_array, low[:, None], high[:, None]).astype(np.float32)
    return raw_array, clipped, names


def inner_fd_centers(
    locations: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    directions: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    offset = radius * sigma[:, None, None, :] * directions[None]
    positive = locations[:, :, None] + offset[:, None]
    negative = locations[:, :, None] - offset[:, None]
    raw = np.concatenate((locations[:, :, None], positive, negative), axis=2)
    clipped = np.clip(raw, low[:, None, None], high[:, None, None]).astype(np.float32)
    return raw.astype(np.float32), clipped


def design_metrics(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks, stable_ranks, conditions = [], [], []
    for value in delta:
        singular = np.linalg.svd(value, compute_uv=False)
        tolerance = max(value.shape) * np.finfo(np.float32).eps * singular[0]
        active = singular[singular > tolerance]
        ranks.append(len(active))
        stable_ranks.append(float(np.sum(singular ** 2) / (singular[0] ** 2 + 1e-12)))
        conditions.append(float(singular[0] / active[-1]) if len(active) else np.inf)
    return (
        np.asarray(ranks, np.int64),
        np.asarray(stable_ranks, np.float32),
        np.asarray(conditions, np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    radii = np.asarray(
        [float(value) for value in args.outer_radii_sigma.split(",")], np.float32
    )
    if len(radii) < 1 or np.any(radii <= 0) or np.max(radii) > 0.075 + 1e-8:
        raise ValueError("outer one-sided radii must be positive and <=0.075 sigma")
    if not 0 < args.fd_radius_sigma <= 0.02:
        raise ValueError("fd-radius-sigma must be in (0,0.02]")
    routing = json.loads(args.routing_manifest.read_text())
    selected_rows = [
        row for row in routing["rows"] if row["selected_for_probe_pilot"]
    ]
    context_index = np.asarray(
        [row["context_index"] for row in selected_rows], np.int64
    )
    if len(context_index) != 100 or len(np.unique(context_index)) != 100:
        raise AssertionError("expected 100 unique routed pilot contexts")

    device = torch.device(args.device)
    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    if not np.all(np.isin(data.episodes[context_index], splits["internal_selection"])):
        raise AssertionError("targeted pilot contains a non-consumed-selection context")
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    actor_input = actor_inputs(tensors, alpha_center, device)
    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    actor = TorchMPPIDeterministicCenterActor(
        maximum_residual_sigma, dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    actor_action, actor_center = residual_outputs(
        actor, actor_input, context_index, args.evaluation_batch_size, device
    )
    with np.load(args.local_labels, allow_pickle=False) as archive:
        source_context = np.asarray(archive["context_index"], np.int64)
        source_center = np.asarray(archive["actor_center"], np.float32)
    source_lookup = {int(value): index for index, value in enumerate(source_context)}
    stored_center = np.asarray(
        [source_center[source_lookup[int(value)]] for value in context_index],
        np.float32,
    )
    actor_center_error = float(np.max(np.abs(actor_center - stored_center)))
    if actor_center_error > 1e-5:
        raise AssertionError(f"frozen Actor center mismatch: {actor_center_error}")

    sigma = data.sigma[context_index].astype(np.float32)
    selected_alpha = alpha_center[context_index].astype(np.float32)
    low, high = bounded_action_limits(
        selected_alpha, sigma, maximum_residual_sigma
    )
    directions, direction_names = response_directions()
    raw_outer, centers, location_names = outer_locations(
        actor_center, sigma, low, high, directions, radii
    )
    fd_directions = hadamard_directions().astype(np.float32)
    raw_inner, inner = inner_fd_centers(
        centers, sigma, low, high, fd_directions, args.fd_radius_sigma
    )
    base_cost = direct_cost(
        selected_alpha, data, tensors, context_index,
        args.evaluation_batch_size, device,
    ).astype(np.float32)

    cost_parts = []
    for start in range(0, len(context_index), args.context_batch_size):
        stop = min(start + args.context_batch_size, len(context_index))
        flat = inner[start:stop].reshape(-1, 8, 2)
        repeated = np.repeat(
            context_index[start:stop], inner.shape[1] * inner.shape[2]
        )
        cost_parts.append(direct_cost(
            flat, data, tensors, repeated,
            args.evaluation_batch_size, device,
        ).reshape(stop - start, inner.shape[1], inner.shape[2]))
        print(json.dumps({
            "contexts_complete": stop,
            "contexts_total": len(context_index),
            "direct_rollouts_complete": int(stop * inner.shape[1] * inner.shape[2]),
        }), flush=True)
    cost = np.concatenate(cost_parts).astype(np.float32)
    z = transformed(base_cost[:, None, None] - cost, args.reward_scale)
    flat_inner = inner.reshape(-1, inner.shape[2], 8, 2)
    flat_alpha = np.repeat(selected_alpha, inner.shape[1], axis=0)
    flat_sigma = np.repeat(sigma, inner.shape[1], axis=0)
    action = centers_to_actions(
        flat_inner, flat_alpha, flat_sigma, maximum_residual_sigma
    )
    location_action = centers_to_actions(
        centers,
        selected_alpha,
        sigma,
        maximum_residual_sigma,
    )
    gradient, curvature, inner_rank = fit_local_parameters(
        action,
        z.reshape(-1, inner.shape[2]),
        location_action.reshape(-1, 8, 2),
    )
    gradient = gradient.reshape(len(context_index), inner.shape[1], 16)
    curvature = curvature.reshape(len(context_index), inner.shape[1])
    inner_rank = inner_rank.reshape(len(context_index), inner.shape[1])
    outer_delta = (
        location_action[:, 1:] - location_action[:, :1]
    ).reshape(len(context_index), inner.shape[1] - 1, 16)
    outer_rank, outer_stable_rank, outer_condition = design_metrics(outer_delta)
    if int(np.min(outer_rank)) < 12:
        raise AssertionError(
            f"outer location design rank below 12: {int(np.min(outer_rank))}"
        )
    if int(np.min(inner_rank)) < 17:
        raise AssertionError(
            f"inner FD design rank below 17: {int(np.min(inner_rank))}"
        )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "targeted_local_response_labels.npz"
    np.savez_compressed(
        arrays_path,
        context_index=context_index,
        episode=data.episodes[context_index],
        pilot_role=np.asarray([row["pilot_role"] for row in selected_rows]),
        stratum_id=np.asarray([row["stratum_id"] for row in selected_rows]),
        reference_speed_mps=np.asarray(
            [row["reference_speed_mps"] for row in selected_rows], np.float32
        ),
        scenario=np.asarray([row["scenario"] for row in selected_rows]),
        clipped=np.asarray([row["clipped"] for row in selected_rows], bool),
        outer_radii_sigma=radii,
        fd_radius_sigma=np.asarray(args.fd_radius_sigma, np.float32),
        response_directions=directions,
        response_direction_names=np.asarray(direction_names),
        fd_directions=fd_directions,
        location_names=np.asarray(location_names),
        sigma=sigma,
        alpha_center=selected_alpha,
        actor_action=actor_action.astype(np.float32),
        actor_center=actor_center.astype(np.float32),
        action_low=low,
        action_high=high,
        raw_outer_centers=raw_outer,
        outer_centers=centers,
        outer_actions=location_action,
        raw_inner_centers=raw_inner,
        inner_centers=inner,
        inner_actions=action.reshape(len(context_index), inner.shape[1], inner.shape[2], 8, 2),
        direct_cost=cost,
        transformed_reward=z,
        local_gradient=gradient,
        local_curvature=curvature,
        inner_design_rank=inner_rank,
        outer_delta_action=outer_delta,
        outer_design_rank=outer_rank,
        outer_stable_rank=outer_stable_rank,
        outer_condition_number=outer_condition,
    )
    rollout_count = int(np.prod(cost.shape))
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "TARGETED_LOCAL_RESPONSE_LABELS_GENERATED_PENDING_VALIDATION",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "local_labels": str(args.local_labels.resolve()),
            "local_labels_sha256": sha256_file(args.local_labels),
            "routing_manifest": str(args.routing_manifest.resolve()),
            "routing_manifest_sha256": sha256_file(args.routing_manifest),
        },
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "counts": {
            "context": len(context_index),
            "target": int(sum(row["pilot_role"] == "target" for row in selected_rows)),
            "matched_easy_control": int(sum(
                row["pilot_role"] == "matched_easy_control" for row in selected_rows
            )),
            "response_direction": len(directions),
            "outer_location": int(centers.shape[1]),
            "inner_fd_center_per_location": int(inner.shape[2]),
            "new_direct_dbm_rollouts": rollout_count,
        },
        "geometry": {
            "outer_radius_semantics": "one-sided center-to-location source sigma RMS",
            "outer_radii_sigma": radii.tolist(),
            "maximum_full_antithetic_chord_sigma": float(2 * np.max(radii)),
            "fd_radius_semantics": "one-sided location-to-FD-center source sigma RMS",
            "fd_radius_sigma": args.fd_radius_sigma,
            "outer_rank_minimum": int(np.min(outer_rank)),
            "outer_stable_rank_median": float(np.median(outer_stable_rank)),
            "outer_condition_median": float(np.median(outer_condition)),
            "outer_condition_maximum": float(np.max(outer_condition)),
            "inner_rank_minimum": int(np.min(inner_rank)),
            "outer_clip_fraction": float(np.mean(raw_outer != centers)),
            "inner_clip_fraction": float(np.mean(raw_inner != inner)),
        },
        "actor_center_max_abs_replay_error": actor_center_error,
        "contract": {
            "actor_frozen": True,
            "analytic_dbm_gradient": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "source_split": "consumed internal-selection mechanism pilot",
            "reward": "asinh((J_alpha-J_center)/5)",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Targeted local-response labels\n\n"
        f"Generated {rollout_count:,} deterministic forward DBM costs with the Actor frozen.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
