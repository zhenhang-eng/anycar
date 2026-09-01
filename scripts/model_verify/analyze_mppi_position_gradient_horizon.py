#!/usr/bin/env python3
"""Horizon-time decomposition of the position-cost action gradient.

Question (review): are position-gradient flips between near-identical
neighbors a genuine whole-rollout geometry difference, or are they
manufactured at a few horizon steps by the cost's structure?

Method:
- For every internal-selection context, autograd the per-step position cost
  w_pos * |traj_t - ref_t|^2 w.r.t. the 8 action knots -> g_pos,t in R^16.
- Position correspondence is fixed time-indexed by construction (no
  nearest-point association anywhere in the cost path).
- For each nearest-neighbor pair (same-episode excluded):
  * signed per-step agreement cos(g_pos,t^A, g_pos,t^B), magnitude-weighted
    opposition mass overall and per horizon third (t<=16, 17-33, 34-50);
  * classification: global reversal (opposing mass >= 0.6) vs localized
    (a single third holds >= 60% of all opposing mass while its own local
    opposition fraction >= 0.5 and the other thirds <= 0.3) vs mixed;
  * horizon-mass profile of |g_pos,t| per context (which third dominates).
- Along/cross-track decomposition of the position error at t = 10/25/50 for
  a detailed example pair, plus early-steering sensitivity maps
  dJ_pos,t/d(steer knots 0..2).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from generate_dbm_direct_gt_validation import interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_local_gradient_critic import cosine_rows
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2/"
    "fresh_fd_audit.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/position_gradient_horizon_20260817_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--example-pair", default="3852,3949")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    fresh = dict(np.load(args.fresh_npz, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    row_by_context = {
        int(row["context_index"]): row for row in manifest["rows"]
    }
    contexts = fresh["context_index"].astype(np.int64)
    count = len(contexts)
    fd_gradient = fresh["gradient"].astype(np.float32)
    episode = fresh["episode"].astype(str)
    weights = data.cost_weights

    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**data.dbm_params)
    )
    horizon = backend.horizon

    per_step = np.zeros((count, horizon, 16), np.float32)
    trajectories = np.zeros((count, horizon, 2), np.float32)
    references = np.zeros((count, horizon, 2), np.float32)
    for position in range(count):
        knots = torch.from_numpy(
            fresh["actor_center"][position].astype(np.float32)
        ).to(device)[None]
        knots.requires_grad_(True)
        actions = interpolate_knots(knots, horizon)
        state = torch.from_numpy(
            data.initial_state_six[contexts[position]].astype(np.float32)
        ).to(device)[None]
        full = backend.rollout_full_state_differentiable(state, actions)
        trajectory = full[0][:, :2]
        if trajectory.shape[0] != horizon:
            trajectory = trajectory[:horizon]
        reference = torch.from_numpy(
            data.direct_reference[contexts[position]].astype(np.float32)
        ).to(device)[:, :2]
        if reference.shape[0] == horizon + 1:
            reference = reference[1:]
        step_costs = weights["position"] * (
            (trajectory - reference) ** 2
        ).sum(dim=1)
        for step in range(horizon):
            gradient = torch.autograd.grad(
                step_costs[step], knots, retain_graph=True
            )[0]
            per_step[position, step] = (
                gradient.detach().cpu().numpy().flatten()
            )
        trajectories[position] = trajectory.detach().cpu().numpy()
        references[position] = reference[:, :2].cpu().numpy()

    step_norm = np.linalg.norm(per_step, axis=2)  # [count, horizon]
    total_position = per_step.sum(axis=1)  # [count, 16]

    features = np.concatenate((
        data.initial_state_six[contexts].astype(np.float32),
        data.direct_reference[contexts].astype(np.float32).reshape(count, -1),
        data.current_action[contexts].astype(np.float32),
        fresh["actor_center"].astype(np.float32).reshape(count, -1),
    ), axis=1)
    quarter = np.quantile(features, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    scale = np.where(scale > 1e-9, scale, np.std(features, axis=0) + 1e-9)
    standardized = features / scale
    distance = np.linalg.norm(
        standardized[:, None, :] - standardized[None, :, :], axis=2
    )
    same = (episode[:, None] == episode[None, :]) | np.eye(count, dtype=bool)
    distance = np.where(same, np.inf, distance)
    nearest = np.argmin(distance, axis=1)

    normalized = fd_gradient / (
        np.linalg.norm(fd_gradient, axis=1, keepdims=True) + 1e-12
    )
    flip = np.asarray([
        float(normalized[i] @ normalized[nearest[i]]) < 0.0
        for i in range(count)
    ])

    thirds = (slice(0, 17), slice(17, 34), slice(34, horizon))
    third_names = ("t1-16", "t17-33", "t34-50")
    classifications = []
    for i in range(count):
        j = nearest[i]
        left = per_step[i]
        right = per_step[j]
        left_norm = step_norm[i]
        right_norm = step_norm[j]
        mass = left_norm * right_norm + 1e-12
        agreement = np.asarray([
            float(left[t] @ right[t]) / (
                np.linalg.norm(left[t]) * np.linalg.norm(right[t]) + 1e-12
            )
            for t in range(horizon)
        ])
        opposing = np.clip(-agreement, 0.0, None) * mass
        opposing_fraction = opposing.sum() / mass.sum()
        third_fractions = []
        third_local = []
        for third in thirds:
            block_mass = mass[third].sum()
            block_opposing = opposing[third].sum()
            third_fractions.append(
                float(block_opposing / (opposing.sum() + 1e-12))
            )
            third_local.append(float(block_opposing / (block_mass + 1e-12)))
        if opposing_fraction >= 0.6:
            label = "global_reversal"
        else:
            concentrated = [
                index for index, value in enumerate(third_fractions)
                if value >= 0.6
            ]
            others_ok = all(
                third_local[index] <= 0.3
                for index in range(3) if index not in concentrated
            )
            if concentrated and third_local[concentrated[0]] >= 0.5 and others_ok:
                label = f"localized_{third_names[concentrated[0]]}"
            else:
                label = "mixed"
        classifications.append({
            "anchor": i,
            "opposing_fraction": float(opposing_fraction),
            "third_fractions": third_fractions,
            "third_local_opposition": third_local,
            "label": label,
        })

    def rate(mask, field=None):
        return float(np.mean(mask)) if mask.sum() else None

    class_counts = {}
    for entry, is_flip in zip(classifications, flip):
        if is_flip:
            class_counts[entry["label"]] = (
                class_counts.get(entry["label"], 0) + 1
            )
    horizon_mass = (step_norm / (
        step_norm.sum(axis=1, keepdims=True) + 1e-12
    ))
    mass_by_third = {
        name: float(np.median(horizon_mass[:, third].sum(axis=1)))
        for name, third in zip(third_names, thirds)
    }
    mass_flip = {
        name: float(np.median(
            horizon_mass[flip][:, third].sum(axis=1)
        ))
        for name, third in zip(third_names, thirds)
    }

    example = [int(v) for v in args.example_pair.split(",")]
    example_report = {}
    for context_id in example:
        index = int(np.flatnonzero(contexts == context_id)[0])
        g_total = total_position[index]
        projection = per_step[index] @ g_total / (
            np.linalg.norm(g_total) + 1e-12
        )
        tangent = references[index, 1] - references[index, 0]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        normal = np.array([-tangent[1], tangent[0]])
        errors = trajectories[index] - references[index]
        along = errors @ tangent
        cross = errors @ normal
        example_report[str(context_id)] = {
            "step_projection_coarse": [
                float(np.sum(projection[third]))
                for third in thirds
            ],
            "early_steering_map_t10": [
                float(v) for v in (
                    per_step[index, 10].reshape(8, 2)[:, 1][:3]
                )
            ],
            "early_steering_map_t30": [
                float(v) for v in (
                    per_step[index, 30].reshape(8, 2)[:, 1][:3]
                )
            ],
            "along_track_error_t10_t25_t50": [
                float(along[t]) for t in (9, 24, horizon - 1)
            ],
            "cross_track_error_t10_t25_t50": [
                float(cross[t]) for t in (9, 24, horizon - 1)
            ],
        }

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "POSITION_GRADIENT_HORIZON_DECOMPOSED_ACTOR_FROZEN",
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "correspondence": (
            "fixed time-indexed trajectory[t] vs reference[t]; no "
            "nearest-point association exists in the cost path"
        ),
        "counts": {
            "contexts": int(count),
            "flipped": int(flip.sum()),
            "same": int((~flip).sum()),
        },
        "flip_classification_counts": class_counts,
        "flip_opposing_fraction_median": float(np.median([
            entry["opposing_fraction"]
            for entry, is_flip in zip(classifications, flip) if is_flip
        ])),
        "same_opposing_fraction_median": float(np.median([
            entry["opposing_fraction"]
            for entry, is_flip in zip(classifications, flip) if not is_flip
        ])),
        "horizon_mass_median_all": mass_by_third,
        "horizon_mass_median_flip_anchors": mass_flip,
        "example_pair": example_report,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    np.savez_compressed(
        args.output_dir / "per_step_position_gradients.npz",
        per_step=per_step,
        step_norm=step_norm,
        nearest_index=nearest,
        flip=flip,
    )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "analysis.json").resolve()),
        "flip_classification_counts": class_counts,
        "flip_opposing_fraction_median": result[
            "flip_opposing_fraction_median"
        ],
        "same_opposing_fraction_median": result[
            "same_opposing_fraction_median"
        ],
        "horizon_mass_median_all": mass_by_third,
        "horizon_mass_median_flip_anchors": mass_flip,
        "example_pair": example_report,
    }, indent=2))


if __name__ == "__main__":
    main()
