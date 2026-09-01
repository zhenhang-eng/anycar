#!/usr/bin/env python3
"""Generate full-rank local proposal-reward labels around the frozen BC center."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    load_config,
    make_controller,
    stable_seeded_noise,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-fullrank-local-reward-v1"
ACTION_DIMENSION = 16
DIRECTION_COUNT = 16
CENTER_COUNT = 1 + 2 * DIRECTION_COUNT
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_local_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_fullrank_diverse_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-seeds", default="23001,23002,23003,23004")
    parser.add_argument("--audit-seeds", default="23101,23102,23103,23104")
    parser.add_argument("--radius-sigma", type=float, default=0.15)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_seeds(text: str, name: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain distinct integer seeds")
    return values


def hadamard_directions() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < ACTION_DIMENSION:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if matrix.shape != (ACTION_DIMENSION, ACTION_DIMENSION):
        raise AssertionError("invalid Hadamard construction")
    gram = matrix @ matrix.T
    if not np.array_equal(gram, ACTION_DIMENSION * np.eye(ACTION_DIMENSION)):
        raise AssertionError("Hadamard directions are not orthogonal")
    return matrix.reshape(DIRECTION_COUNT, 8, 2)


def center_names() -> tuple[str, ...]:
    names = ["network"]
    for index in range(DIRECTION_COUNT):
        names.extend((f"direction_{index:02d}_pos", f"direction_{index:02d}_neg"))
    return tuple(names)


def make_centers(
    base: np.ndarray,
    directions: np.ndarray,
    sigma: np.ndarray,
    radius_sigma: float,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    perturbation = (
        float(radius_sigma) * directions * sigma.reshape(1, 1, 2)
    ).astype(np.float32)
    centers = [np.asarray(base, dtype=np.float32)]
    for value in perturbation:
        centers.extend((base + value, base - value))
    raw = np.asarray(centers, dtype=np.float32)
    clipped = np.clip(raw, action_min, action_max).astype(np.float32)
    return raw, clipped


def fully_batched_evaluation(
    centers: np.ndarray,
    config: dict[str, Any],
    controller: Any,
    backend: Any,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    warm: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    evaluation_seeds: list[int],
) -> dict[str, np.ndarray]:
    """Evaluate all center/seed/sample combinations in one DBM batch."""
    sample_count = int(config["proposal_evaluation"]["num_samples"])
    proposal_sigma = sigma * float(config["proposal_evaluation"]["sigma_scale"])
    seed_count = len(evaluation_seeds)
    center_count = len(centers)
    noises = np.asarray(
        [
            stable_seeded_noise(
                seed, sample_count, tuple(centers.shape[1:]), proposal_sigma
            )
            for seed in evaluation_seeds
        ],
        dtype=np.float32,
    )
    raw = centers[None, :, None] + noises[:, None]
    knots = np.clip(raw, action_min, action_max).astype(np.float32)
    flat_count = seed_count * center_count * sample_count
    cost, actions, _ = evaluate_knots(
        controller,
        backend,
        knots.reshape(flat_count, 8, 2),
        history,
        initial_state,
        current_action,
        reference,
    )
    cost = cost.reshape(seed_count, center_count, sample_count)
    actions = actions.reshape(
        seed_count, center_count, sample_count, *actions.shape[1:]
    )
    temperature = float(config["objective"]["temperature"])
    shifted = cost.astype(np.float64) - cost.min(axis=2, keepdims=True)
    unnormalized = np.exp(-shifted / temperature)
    weight = unnormalized / unnormalized.sum(axis=2, keepdims=True)
    weighted_actions = np.sum(
        weight[..., None, None] * actions, axis=2
    ).astype(np.float32)
    output_cost, _ = evaluate_actions(
        controller,
        backend,
        weighted_actions.reshape(
            seed_count * center_count, *weighted_actions.shape[2:]
        ),
        history,
        initial_state,
        current_action,
        reference,
    )
    output_cost = output_cost.reshape(seed_count, center_count).T
    best = cost.min(axis=2).T.astype(np.float32)
    p10 = np.quantile(cost, 0.10, axis=2).T.astype(np.float32)
    median = np.median(cost, axis=2).T.astype(np.float32)
    softmin = (
        cost.min(axis=2)
        - temperature * np.log(np.mean(np.exp(-shifted / temperature), axis=2))
    ).T.astype(np.float32)
    ess = (1.0 / np.sum(np.square(weight), axis=2)).T.astype(np.float32)
    clip = np.mean(raw != knots, axis=(2, 3, 4)).T.astype(np.float32)
    return {
        "proposal_weighted_output_cost": output_cost.astype(np.float32),
        "proposal_best_cost": best,
        "proposal_p10_cost": p10,
        "proposal_median_cost": median,
        "proposal_softmin_cost": softmin,
        "proposal_effective_sample_size": ess,
        "proposal_clip_fraction": clip,
    }


def process_one(
    parent_path: Path,
    source_root: Path,
    config: dict[str, Any],
    directions: np.ndarray,
    radius_sigma: float,
    selection_seeds: list[int],
    audit_seeds: list[int],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episode_id = parent_path.parent.name
    source_path = source_root / episode_id / "snapshots" / parent_path.name
    source_hash = sha256_file(source_path)
    parent_hash = sha256_file(parent_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(
        parent_path, allow_pickle=False
    ) as parent:
        if str(parent["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{parent_path}: parent/source hash mismatch")
        base = np.asarray(parent["network_center_knots"], dtype=np.float32)
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
        action_min = np.asarray(params["action_min"], dtype=np.float32)
        action_max = np.asarray(params["action_max"], dtype=np.float32)
        raw_centers, centers = make_centers(
            base,
            directions,
            sigma,
            radius_sigma,
            action_min,
            action_max,
        )
        controller, backend = make_controller(source, config, device)
        common = dict(
            centers=centers,
            config=config,
            controller=controller,
            backend=backend,
            history=torch.from_numpy(source["history"]).to(device),
            initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(
                1, 5
            ),
            current_action=torch.from_numpy(source["current_action"]).to(device).reshape(
                1, 2
            ),
            reference=controller._prepare_reference(source["reference"]),
            warm=np.asarray(source["sampling_mean_knots"], dtype=np.float32),
            sigma=sigma,
            action_min=action_min,
            action_max=action_max,
        )
        selection = fully_batched_evaluation(
            **common, evaluation_seeds=selection_seeds
        )
        audit = fully_batched_evaluation(**common, evaluation_seeds=audit_seeds)
    standardized = (centers - base[None]) / sigma.reshape(1, 1, 2)
    local_rank = int(np.linalg.matrix_rank(standardized.reshape(CENTER_COUNT, -1)))
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "source_snapshot_sha256": np.asarray(source_hash),
        "parent_label_sha256": np.asarray(parent_hash),
        "center_names": np.asarray(center_names()),
        "base_center_knots": base,
        "normalized_directions": directions,
        "radius_sigma": np.asarray(radius_sigma, dtype=np.float32),
        "raw_centers": raw_centers,
        "centers": centers,
        "local_direction_rank": np.asarray(local_rank, dtype=np.int32),
        "proposal_evaluation_seeds": np.asarray(selection_seeds, dtype=np.int64),
        **selection,
        "audit_evaluation_seeds": np.asarray(audit_seeds, dtype=np.int64),
        **{f"audit_{name}": value for name, value in audit.items()},
    }
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator_id": GENERATOR_ID,
        "episode_id": episode_id,
        "control_step": int(parent_path.stem.removeprefix("step_")),
        "source_snapshot_sha256": source_hash,
        "parent_label_sha256": parent_hash,
        "center_count": CENTER_COUNT,
        "direction_count": DIRECTION_COUNT,
        "local_direction_rank": local_rank,
        "radius_sigma": radius_sigma,
        "selection_seeds": selection_seeds,
        "audit_seeds": audit_seeds,
    }
    return arrays, metadata


def summarize_costs(
    selection: np.ndarray, audit: np.ndarray
) -> dict[str, Any]:
    # [state, center, seed], with center 0 as the frozen BC base.
    selection_mean = selection.mean(axis=2)
    audit_mean = audit.mean(axis=2)
    selection_slopes = []
    audit_slopes = []
    for direction in range(DIRECTION_COUNT):
        positive = 1 + 2 * direction
        negative = positive + 1
        selection_slopes.append(selection_mean[:, positive] - selection_mean[:, negative])
        audit_slopes.append(audit_mean[:, positive] - audit_mean[:, negative])
    selection_slopes = np.stack(selection_slopes, axis=1)
    audit_slopes = np.stack(audit_slopes, axis=1)
    valid = (np.abs(selection_slopes) > 0.1) & (np.abs(audit_slopes) > 0.1)
    sign_agreement = np.sign(selection_slopes[valid]) == np.sign(audit_slopes[valid])
    per_direction = []
    for direction in range(DIRECTION_COUNT):
        mask = valid[:, direction]
        per_direction.append(
            {
                "direction": direction,
                "valid_count": int(mask.sum()),
                "sign_stability": float(
                    np.mean(
                        np.sign(selection_slopes[mask, direction])
                        == np.sign(audit_slopes[mask, direction])
                    )
                )
                if mask.any()
                else 0.0,
                "selection_audit_correlation": float(
                    np.corrcoef(
                        selection_slopes[:, direction], audit_slopes[:, direction]
                    )[0, 1]
                ),
            }
        )
    return {
        "base_cost_mean": {
            "selection": float(selection_mean[:, 0].mean()),
            "audit": float(audit_mean[:, 0].mean()),
        },
        "selection_audit_center_cost_mae": float(
            np.mean(np.abs(selection_mean - audit_mean))
        ),
        "directional_sign_stability": float(np.mean(sign_agreement)),
        "directional_valid_comparison_count": int(valid.sum()),
        "directional_difference_mae": float(
            np.mean(np.abs(selection_slopes - audit_slopes))
        ),
        "per_direction": per_direction,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.radius_sigma <= 0.5:
        raise ValueError("--radius-sigma must be in (0,0.5]")
    selection_seeds = parse_seeds(args.selection_seeds, "selection seeds")
    audit_seeds = parse_seeds(args.audit_seeds, "audit seeds")
    if set(selection_seeds) & set(audit_seeds):
        raise ValueError("selection and audit seeds overlap")
    source_root = args.source.resolve()
    parent_root = args.parent_labels.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    config = load_config(parent_root / "teacher_config.json")
    forbidden_seeds = set(config["proposal_evaluation"]["seeds"]) | set(
        config["proposal_evaluation"]["audit_seeds"]
    )
    forbidden_seeds |= set(parent_summary["selection_seeds"]) | set(
        parent_summary["audit_seeds"]
    )
    if forbidden_seeds & (set(selection_seeds) | set(audit_seeds)):
        raise ValueError("full-rank seeds overlap existing T1/local seeds")
    paths = sorted(parent_root.glob("episode_*/*.npz"))
    if args.max_snapshots > 0:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise FileNotFoundError(f"no parent labels under {parent_root}")
    directions = hadamard_directions()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    device = torch.device(args.device)
    started = time.perf_counter()
    selection_values = []
    audit_values = []
    ranks = []
    clip_values = []
    try:
        for index, parent_path in enumerate(paths, start=1):
            arrays, metadata = process_one(
                parent_path,
                source_root,
                config,
                directions,
                args.radius_sigma,
                selection_seeds,
                audit_seeds,
                device,
            )
            episode_dir = staging / parent_path.parent.name
            episode_dir.mkdir(exist_ok=True)
            target = episode_dir / parent_path.name
            np.savez_compressed(target, **arrays)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            selection_values.append(arrays["proposal_weighted_output_cost"])
            audit_values.append(arrays["audit_proposal_weighted_output_cost"])
            ranks.append(int(arrays["local_direction_rank"]))
            clip_values.append(np.mean(arrays["raw_centers"] != arrays["centers"]))
            if index % 20 == 0 or index == len(paths):
                elapsed = time.perf_counter() - started
                print(
                    f"[{index:04d}/{len(paths):04d}] full-rank labels "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
        shutil.copy2(parent_root / "splits.json", staging / "splits.json")
        shutil.copy2(parent_root / "teacher_config.json", staging / "teacher_config.json")
        selection_array = np.asarray(selection_values, dtype=np.float32)
        audit_array = np.asarray(audit_values, dtype=np.float32)
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root),
            "parent_labels": str(parent_root),
            "snapshot_count": len(paths),
            "action_dimension": ACTION_DIMENSION,
            "direction_count": DIRECTION_COUNT,
            "center_count": CENTER_COUNT,
            "direction_design": "16x16 Sylvester Hadamard, antithetic pairs",
            "radius_sigma": args.radius_sigma,
            "selection_seeds": selection_seeds,
            "audit_seeds": audit_seeds,
            "candidate_count_per_center_seed": int(
                config["proposal_evaluation"]["num_samples"]
            ),
            "local_rank": {
                "minimum": int(np.min(ranks)),
                "maximum": int(np.max(ranks)),
                "full_rank_count": int(np.sum(np.asarray(ranks) == ACTION_DIMENSION)),
            },
            "center_clip_fraction": {
                "mean": float(np.mean(clip_values)),
                "maximum": float(np.max(clip_values)),
            },
            "reward_diagnostics": summarize_costs(selection_array, audit_array),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "semantics": (
                "No new scenario states are collected. Each label evaluates a full-rank "
                "antithetic neighborhood around the frozen BC center with fixed DBM MPPI "
                "weighted-output reward and disjoint selection/audit noise seeds."
            ),
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
