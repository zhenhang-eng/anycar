#!/usr/bin/env python3
"""Relabel the local BC-to-teacher proposal region with fixed-DBM rollouts.

This is not a new teacher search.  It evaluates a deterministic bank containing
warm, the frozen BC actor output, the T1 teacher, interpolation points, and small
antithetic perturbations around the BC output.  Common random numbers are used
within every snapshot; disjoint selection and audit seeds are stored separately.
"""

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

from evaluate_mppi_proposal_bc import load_policy, predict_center
from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    load_config,
    make_controller,
    stable_seeded_noise,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-proposal-critic-local-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_local_diverse_20260805_v1"
)
CENTER_NAMES = (
    "warm",
    "network",
    "teacher",
    "warm_network_050",
    "network_teacher_025",
    "network_teacher_050",
    "network_teacher_075",
    "network_local_pos0",
    "network_local_neg0",
    "network_local_pos1",
    "network_local_neg1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-seeds", default="21001,21002")
    parser.add_argument("--audit-seeds", default="21101,21102")
    parser.add_argument("--local-sigma-scale", type=float, default=0.15)
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


def smooth_direction(direction: np.ndarray) -> np.ndarray:
    padded = np.pad(direction, ((1, 1), (0, 0)), mode="edge")
    return (
        0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    ).astype(np.float32)


def local_directions(
    episode_id: str, control_step: int, sigma: np.ndarray
) -> np.ndarray:
    episode_number = int(episode_id.rsplit("_", 1)[1])
    seed = 31_000_000 + episode_number * 100_000 + control_step
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2):
        direction = smooth_direction(rng.standard_normal((8, 2)).astype(np.float32))
        rms = np.sqrt(np.mean(np.square(direction), axis=0, keepdims=True))
        direction /= np.maximum(rms, 1e-6)
        values.append(direction * sigma.reshape(1, 2))
    return np.asarray(values, dtype=np.float32)


def make_centers(
    warm: np.ndarray,
    network: np.ndarray,
    teacher: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    episode_id: str,
    control_step: int,
    local_sigma_scale: float,
) -> np.ndarray:
    def lerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
        return left + fraction * (right - left)

    directions = local_directions(episode_id, control_step, sigma)
    local = directions * float(local_sigma_scale)
    centers = np.asarray(
        (
            warm,
            network,
            teacher,
            lerp(warm, network, 0.50),
            lerp(network, teacher, 0.25),
            lerp(network, teacher, 0.50),
            lerp(network, teacher, 0.75),
            network + local[0],
            network - local[0],
            network + local[1],
            network - local[1],
        ),
        dtype=np.float32,
    )
    return np.clip(centers, action_min, action_max).astype(np.float32)


def discover(t1_root: Path, max_snapshots: int) -> list[Path]:
    paths = sorted(t1_root.glob("episode_*/*.npz"))
    if max_snapshots > 0:
        paths = paths[:max_snapshots]
    if not paths:
        raise FileNotFoundError(f"no T1 labels under {t1_root}")
    return paths


def batched_proposal_evaluation(
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
    """Vectorized equivalent of the reduced proposal_evaluation outputs."""
    proposal = config["proposal_evaluation"]
    sample_count = int(proposal["num_samples"])
    proposal_sigma = sigma * float(proposal["sigma_scale"])
    center_count = len(centers)
    seed_count = len(evaluation_seeds)
    shape = (center_count, seed_count)
    weighted_cost = np.empty(shape, dtype=np.float32)
    best_cost = np.empty(shape, dtype=np.float32)
    p10_cost = np.empty(shape, dtype=np.float32)
    median_cost = np.empty(shape, dtype=np.float32)
    softmin_cost = np.empty(shape, dtype=np.float32)
    ess = np.empty(shape, dtype=np.float32)
    clip_fraction = np.empty(shape, dtype=np.float32)
    temperature = float(config["objective"]["temperature"])
    for seed_index, seed in enumerate(evaluation_seeds):
        noise = stable_seeded_noise(
            seed, sample_count, tuple(centers.shape[1:]), proposal_sigma
        )
        raw = centers[:, None] + noise[None]
        knots = np.clip(raw, action_min, action_max).astype(np.float32)
        cost, actions, _ = evaluate_knots(
            controller,
            backend,
            knots.reshape(center_count * sample_count, 8, 2),
            history,
            initial_state,
            current_action,
            reference,
        )
        cost = cost.reshape(center_count, sample_count)
        actions = actions.reshape(center_count, sample_count, *actions.shape[1:])
        shifted = cost.astype(np.float64) - cost.min(axis=1, keepdims=True)
        unnormalized = np.exp(-shifted / temperature)
        weight = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        weighted_actions = np.sum(weight[..., None, None] * actions, axis=1).astype(
            np.float32
        )
        output_cost, _ = evaluate_actions(
            controller,
            backend,
            weighted_actions,
            history,
            initial_state,
            current_action,
            reference,
        )
        weighted_cost[:, seed_index] = output_cost
        best_cost[:, seed_index] = cost.min(axis=1)
        p10_cost[:, seed_index] = np.quantile(cost, 0.10, axis=1)
        median_cost[:, seed_index] = np.median(cost, axis=1)
        softmin_cost[:, seed_index] = cost.min(axis=1) - temperature * np.log(
            np.mean(np.exp(-shifted / temperature), axis=1)
        )
        ess[:, seed_index] = 1.0 / np.sum(np.square(weight), axis=1)
        clip_fraction[:, seed_index] = np.mean(raw != knots, axis=(1, 2, 3))
    shift_rms = np.sqrt(
        np.mean(
            np.square((centers - warm[None]) / sigma.reshape(1, 1, 2)),
            axis=(1, 2),
        )
    ).astype(np.float32)
    boundary_fraction = np.mean(
        (centers <= action_min.reshape(1, 1, 2) + 1e-6)
        | (centers >= action_max.reshape(1, 1, 2) - 1e-6),
        axis=(1, 2),
    ).astype(np.float32)
    components = {
        "weighted_output_cost_mean": weighted_cost.mean(1),
        "weighted_output_cost_std": weighted_cost.std(1),
        "p10_cost_mean": p10_cost.mean(1),
        "softmin_cost_mean": softmin_cost.mean(1),
        "center_shift_standardized_rms": shift_rms,
        "boundary_fraction": boundary_fraction,
    }
    score = sum(
        float(config["selection_score"][name]) * value
        for name, value in components.items()
    )
    return {
        "proposal_weighted_output_cost": weighted_cost,
        "proposal_best_cost": best_cost,
        "proposal_p10_cost": p10_cost,
        "proposal_median_cost": median_cost,
        "proposal_softmin_cost": softmin_cost,
        "proposal_effective_sample_size": ess,
        "proposal_clip_fraction": clip_fraction,
        "center_shift_standardized_rms": shift_rms,
        "center_boundary_fraction": boundary_fraction,
        "selection_score": np.asarray(score, dtype=np.float32),
    }


def process_one(
    label_path: Path,
    source_root: Path,
    actor: torch.nn.Module,
    normalization: Any,
    config: dict[str, Any],
    selection_seeds: list[int],
    audit_seeds: list[int],
    local_sigma_scale: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episode_id = label_path.parent.name
    source_path = source_root / episode_id / "snapshots" / label_path.name
    source_hash = sha256_file(source_path)
    t1_hash = sha256_file(label_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(
        label_path, allow_pickle=False
    ) as t1:
        predicted_delta, network = predict_center(
            actor, normalization, source, device
        )
        warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
        teacher = np.asarray(t1["teacher_center_knots"], dtype=np.float32)
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
        action_min = np.asarray(params["action_min"], dtype=np.float32)
        action_max = np.asarray(params["action_max"], dtype=np.float32)
        control_step = int(source["control_step"])
        centers = make_centers(
            warm,
            network,
            teacher,
            sigma,
            action_min,
            action_max,
            episode_id,
            control_step,
            local_sigma_scale,
        )
        controller, backend = make_controller(source, config, device)
        history = torch.from_numpy(source["history"]).to(device)
        state = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
        current_action = torch.from_numpy(source["current_action"]).to(device).reshape(
            1, 2
        )
        reference = controller._prepare_reference(source["reference"])
        common = dict(
            centers=centers,
            config=config,
            controller=controller,
            backend=backend,
            history=history,
            initial_state=state,
            current_action=current_action,
            reference=reference,
            warm=warm,
            sigma=sigma,
            action_min=action_min,
            action_max=action_max,
        )
        selection = batched_proposal_evaluation(
            **common, evaluation_seeds=selection_seeds
        )
        audit = batched_proposal_evaluation(
            **common, evaluation_seeds=audit_seeds
        )
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "source_snapshot_sha256": np.asarray(source_hash),
        "t1_label_sha256": np.asarray(t1_hash),
        "center_names": np.asarray(CENTER_NAMES),
        "shortlist_center_names": np.asarray(CENTER_NAMES),
        "shortlist_centers": centers,
        "proposal_evaluation_seeds": np.asarray(selection_seeds, dtype=np.int64),
        **selection,
        "audit_evaluation_seeds": np.asarray(audit_seeds, dtype=np.int64),
        **{f"audit_{name}": value for name, value in audit.items()},
        "warm_shortlist_index": np.asarray(0, dtype=np.int32),
        "network_shortlist_index": np.asarray(1, dtype=np.int32),
        "teacher_shortlist_index": np.asarray(2, dtype=np.int32),
        "network_center_knots": centers[1],
        "network_delta_knots": predicted_delta.astype(np.float32),
        "teacher_center_knots": centers[2],
        "teacher_delta_knots": (centers[2] - centers[0]).astype(np.float32),
    }
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator_id": GENERATOR_ID,
        "episode_id": episode_id,
        "control_step": control_step,
        "source_snapshot": str(source_path.relative_to(source_root)),
        "source_snapshot_sha256": source_hash,
        "t1_label_sha256": t1_hash,
        "center_names": list(CENTER_NAMES),
        "selection_seeds": selection_seeds,
        "audit_seeds": audit_seeds,
        "local_sigma_scale": local_sigma_scale,
    }
    return arrays, metadata


def aggregate_cost(label_paths: list[Path], prefix: str = "") -> dict[str, Any]:
    costs = []
    for path in label_paths:
        with np.load(path, allow_pickle=False) as data:
            costs.append(np.asarray(data[f"{prefix}proposal_weighted_output_cost"]).mean(1))
    values = np.asarray(costs)
    warm = values[:, 0]
    return {
        "snapshot_count": len(values),
        "center_names": list(CENTER_NAMES),
        "mean_cost": dict(zip(CENTER_NAMES, values.mean(0).tolist())),
        "mean_gain_from_warm": dict(
            zip(CENTER_NAMES, (warm[:, None] - values).mean(0).tolist())
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.local_sigma_scale <= 1.0:
        raise ValueError("--local-sigma-scale must be in (0,1]")
    selection_seeds = parse_seeds(args.selection_seeds, "selection seeds")
    audit_seeds = parse_seeds(args.audit_seeds, "audit seeds")
    if set(selection_seeds) & set(audit_seeds):
        raise ValueError("selection and audit seeds must be disjoint")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_root = args.source.resolve()
    t1_root = args.t1_labels.resolve()
    actor_path = args.actor_checkpoint.resolve()
    config = load_config(t1_root / "teacher_config.json")
    old_seeds = set(config["proposal_evaluation"]["seeds"]) | set(
        config["proposal_evaluation"]["audit_seeds"]
    )
    if old_seeds & (set(selection_seeds) | set(audit_seeds)):
        raise ValueError("local relabel seeds must be disjoint from T1 seeds")
    paths = discover(t1_root, args.max_snapshots)
    device = torch.device(args.device)
    actor, normalization, _ = load_policy(actor_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    started = time.perf_counter()
    try:
        written = []
        for index, label_path in enumerate(paths, start=1):
            arrays, metadata = process_one(
                label_path,
                source_root,
                actor,
                normalization,
                config,
                selection_seeds,
                audit_seeds,
                args.local_sigma_scale,
                device,
            )
            episode_dir = staging / label_path.parent.name
            episode_dir.mkdir(exist_ok=True)
            target = episode_dir / label_path.name
            np.savez_compressed(target, **arrays)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            written.append(target)
            if index % 20 == 0 or index == len(paths):
                print(f"[{index:04d}/{len(paths):04d}] local proposal labels")
        shutil.copy2(t1_root / "splits.json", staging / "splits.json")
        shutil.copy2(t1_root / "teacher_config.json", staging / "teacher_config.json")
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root),
            "t1_labels": str(t1_root),
            "actor_checkpoint": str(actor_path),
            "actor_checkpoint_sha256": sha256_file(actor_path),
            "snapshot_count": len(written),
            "center_names": list(CENTER_NAMES),
            "selection_seeds": selection_seeds,
            "audit_seeds": audit_seeds,
            "candidate_count_per_center_seed": int(
                config["proposal_evaluation"]["num_samples"]
            ),
            "local_sigma_scale": args.local_sigma_scale,
            "selection": aggregate_cost(written),
            "audit": aggregate_cost(written, "audit_"),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "semantics": (
                "Local critic supervision only; centers are deterministic functions of "
                "warm, frozen BC, T1 teacher, and stable snapshot-local perturbations."
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
