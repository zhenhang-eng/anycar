#!/usr/bin/env python3
"""Relabel critic-visited two-pass centers with repeated second-pass rewards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIFeedbackQuadraticCritic,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    load_config,
    make_controller,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from generate_dbm_two_pass_feedback_labels import antithetic_noise


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-two-pass-risk-replay-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_CRITIC = Path("outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1")
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
RADII = (0.03, 0.06, 0.10, 0.15)
CENTER_COUNT = 1 + 2 * len(RADII)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radii", default=",".join(map(str, RADII)))
    parser.add_argument(
        "--selection-seeds", default="26001,26002,26003,26004,26005,26006,26007,26008"
    )
    parser.add_argument(
        "--audit-seeds", default="26101,26102,26103,26104,26105,26106,26107,26108"
    )
    parser.add_argument("--samples-per-center-seed", type=int, default=64)
    parser.add_argument("--second-noise-scale", type=float, default=0.10)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seed list must contain distinct integers")
    return values


def parse_floats(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or values != sorted(set(values)) or values[0] <= 0 or values[-1] > 0.25:
        raise ValueError("radii must be sorted distinct values in (0,0.25]")
    return values


def center_names(radii: list[float]) -> tuple[str, ...]:
    names = ["guided_anchor"]
    for radius in radii:
        token = f"{radius:.3f}".replace(".", "p")
        names.extend((f"critic_pos_{token}", f"critic_neg_{token}"))
    return tuple(names)


def load_critic_ensemble(
    critic_dir: Path, device: torch.device
) -> tuple[
    list[TorchMPPIFeedbackQuadraticCritic],
    MPPIProposalNormalization,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    summary = json.loads((critic_dir / "training_summary.json").read_text())
    checkpoints = [Path(value) for value in summary["ensemble_checkpoints"]]
    models = []
    loaded = []
    normalization = None
    feedback_mean = feedback_std = None
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        model = TorchMPPIFeedbackQuadraticCritic().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        loaded.append(str(path.resolve()))
        if normalization is None:
            normalization = MPPIProposalNormalization.from_dict(
                checkpoint["state_normalization"]
            )
            feedback_mean = np.asarray(checkpoint["feedback_mean"], np.float32)
            feedback_std = np.asarray(checkpoint["feedback_std"], np.float32)
    if normalization is None or feedback_mean is None or feedback_std is None:
        raise ValueError("empty critic ensemble")
    return models, normalization, feedback_mean, feedback_std, loaded


@torch.no_grad()
def critic_directions(
    models: list[TorchMPPIFeedbackQuadraticCritic],
    normalization: MPPIProposalNormalization,
    feedback_mean: np.ndarray,
    feedback_std: np.ndarray,
    source: np.lib.npyio.NpzFile,
    anchors: np.ndarray,
    feedback: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repeat_count = len(anchors)
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.repeat(np.asarray(source["history"], np.float32), repeat_count, axis=0)
    reference_one = ego_reference_features(source["reference_ego"], float(state[3]))
    reference = np.repeat(reference_one[None], repeat_count, axis=0)
    current = np.repeat(
        np.asarray((state[3], state[4], *action), np.float32)[None],
        repeat_count,
        axis=0,
    )
    history, reference, current = normalization.normalize_numpy(
        history, reference, current
    )
    tensors = (
        torch.from_numpy(history).to(device),
        torch.from_numpy(reference).to(device),
        torch.from_numpy(current).to(device),
        torch.from_numpy(anchors).to(device),
        torch.from_numpy((feedback - feedback_mean) / feedback_std).to(device),
    )
    gradients = torch.stack(
        [model.local_parameters(*tensors)[0].flatten(1) for model in models]
    )
    mean = gradients.mean(dim=0).cpu().numpy().astype(np.float32)
    standard_deviation = gradients.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    maximum = np.max(np.abs(mean), axis=1, keepdims=True)
    direction = mean / np.maximum(maximum, 1e-6)
    return direction.reshape(-1, 8, 2), mean, standard_deviation


def make_center_bank(
    anchors: np.ndarray,
    directions: np.ndarray,
    radii: list[float],
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    banks = []
    for anchor, direction in zip(anchors, directions):
        centers = [anchor]
        for radius in radii:
            delta = float(radius) * direction * sigma.reshape(1, 2)
            centers.extend((anchor + delta, anchor - delta))
        banks.append(centers)
    raw = np.asarray(banks, np.float32)
    return raw, np.clip(raw, action_min, action_max).astype(np.float32)


def evaluate_center_bank(
    centers: np.ndarray,
    seeds: list[int],
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
    repeat_count, center_count = centers.shape[:2]
    noise = np.asarray(
        [
            antithetic_noise(seed, sample_count, (8, 2), local_sigma)
            for seed in seeds
        ],
        np.float32,
    )
    raw = centers[:, None, :, None] + noise[None, :, None]
    knots = np.clip(raw, action_min, action_max).astype(np.float32)
    cost, actions, _ = evaluate_knots(
        controller,
        backend,
        knots.reshape(-1, 8, 2),
        history,
        initial_state,
        current_action,
        reference,
    )
    seed_count = len(seeds)
    cost = cost.reshape(repeat_count, seed_count, center_count, sample_count)
    actions = actions.reshape(
        repeat_count, seed_count, center_count, sample_count, *actions.shape[1:]
    )
    shifted = cost.astype(np.float64) - cost.min(axis=3, keepdims=True)
    unnormalized = np.exp(-shifted / temperature)
    weight = unnormalized / unnormalized.sum(axis=3, keepdims=True)
    weighted_actions = np.sum(weight[..., None, None] * actions, axis=3).astype(np.float32)
    output_cost, _ = evaluate_actions(
        controller,
        backend,
        weighted_actions.reshape(-1, *weighted_actions.shape[3:]),
        history,
        initial_state,
        current_action,
        reference,
    )
    output_cost = output_cost.reshape(repeat_count, seed_count, center_count).transpose(0, 2, 1)
    advantage = output_cost[:, :1, :] - output_cost
    return {
        "proposal_output_cost_by_seed": output_cost.astype(np.float32),
        "paired_advantage_by_seed": advantage.astype(np.float32),
        "paired_advantage_mean": advantage.mean(axis=2).astype(np.float32),
        "paired_advantage_std": advantage.std(axis=2).astype(np.float32),
        "paired_advantage_p10": np.quantile(advantage, 0.10, axis=2).astype(np.float32),
        "paired_win_probability": np.mean(advantage > 0.0, axis=2).astype(np.float32),
        "effective_sample_size_by_seed": (
            1.0 / np.sum(weight * weight, axis=3)
        ).transpose(0, 2, 1).astype(np.float32),
        "candidate_clip_fraction_by_seed": np.mean(
            raw != knots, axis=(3, 4, 5)
        ).transpose(0, 2, 1).astype(np.float32),
    }


def process_one(
    parent_path: Path,
    source_root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    radii: list[float],
    selection_seeds: list[int],
    audit_seeds: list[int],
    models: list[TorchMPPIFeedbackQuadraticCritic],
    normalization: MPPIProposalNormalization,
    feedback_mean: np.ndarray,
    feedback_std: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episode_id = parent_path.parent.name
    source_path = source_root / episode_id / "snapshots" / parent_path.name
    source_hash, parent_hash = sha256_file(source_path), sha256_file(parent_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(
        parent_path, allow_pickle=False
    ) as parent:
        if str(parent["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{parent_path}: source hash mismatch")
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], np.float32)
        action_min = np.asarray(params["action_min"], np.float32)
        action_max = np.asarray(params["action_max"], np.float32)
        controller, backend = make_controller(source, config, device)
        common_rollout = dict(
            sample_count=args.samples_per_center_seed,
            local_sigma=sigma * args.second_noise_scale,
            action_min=action_min,
            action_max=action_max,
            temperature=float(config["objective"]["temperature"]),
            controller=controller,
            backend=backend,
            history=torch.from_numpy(source["history"]).to(device),
            initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
            current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
            reference=controller._prepare_reference(source["reference"]),
        )
        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION, np.int32),
            "source_snapshot_sha256": np.asarray(source_hash),
            "parent_label_sha256": np.asarray(parent_hash),
            "center_names": np.asarray(center_names(radii)),
            "radii_sigma": np.asarray(radii, np.float32),
        }
        for prefix, seeds in (("", selection_seeds), ("audit_", audit_seeds)):
            parent_prefix = "" if not prefix else "audit_"
            anchors = np.asarray(parent[f"{parent_prefix}guided_center_knots"], np.float32)
            feedback = np.asarray(parent[f"{parent_prefix}first_pass_feedback"], np.float32)
            direction, gradient_mean, gradient_std = critic_directions(
                models,
                normalization,
                feedback_mean,
                feedback_std,
                source,
                anchors,
                feedback,
                device,
            )
            raw_centers, centers = make_center_bank(
                anchors, direction, radii, sigma, action_min, action_max
            )
            result = evaluate_center_bank(
                centers=centers, seeds=seeds, **common_rollout
            )
            arrays.update(
                {
                    f"{prefix}first_pass_seed": np.asarray(
                        parent[f"{parent_prefix}first_pass_seed"], np.int64
                    ),
                    f"{prefix}feedback_sha256": np.asarray(
                        [hashlib.sha256(value.tobytes()).hexdigest() for value in feedback]
                    ),
                    f"{prefix}guided_center_knots": anchors,
                    f"{prefix}critic_gradient_mean": gradient_mean,
                    f"{prefix}critic_gradient_std": gradient_std,
                    f"{prefix}normalized_critic_direction": direction,
                    f"{prefix}raw_centers": raw_centers,
                    f"{prefix}centers": centers,
                    f"{prefix}evaluation_seeds": np.asarray(seeds, np.int64),
                    **{f"{prefix}{name}": value for name, value in result.items()},
                }
            )
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator_id": GENERATOR_ID,
        "episode_id": episode_id,
        "control_step": int(parent_path.stem.removeprefix("step_")),
        "source_snapshot_sha256": source_hash,
        "parent_label_sha256": parent_hash,
        "selection_seeds": selection_seeds,
        "audit_seeds": audit_seeds,
    }
    return arrays, metadata


def aggregate_summary(values: np.ndarray) -> dict[str, Any]:
    # [state, repeat, center, seed]
    advantage = values[:, :, 0:1] - values
    mean = advantage.mean(axis=3)
    return {
        "per_center_mean_advantage": mean.mean(axis=(0, 1)).tolist(),
        "per_center_median_advantage": np.median(mean, axis=(0, 1)).tolist(),
        "per_center_positive_mean_fraction": np.mean(mean > 0, axis=(0, 1)).tolist(),
        "per_center_all_seed_win_fraction": np.mean(
            np.all(advantage > 0, axis=3), axis=(0, 1)
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    radii = parse_floats(args.radii)
    selection_seeds, audit_seeds = parse_ints(args.selection_seeds), parse_ints(args.audit_seeds)
    if set(selection_seeds) & set(audit_seeds):
        raise ValueError("selection and audit reward seeds overlap")
    if len(selection_seeds) != len(audit_seeds):
        raise ValueError("selection and audit seed counts must match")
    source_root, parent_root = args.source.resolve(), args.parent_labels.resolve()
    critic_dir, output = args.critic_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(parent_root / "teacher_config.json")
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    forbidden_seeds = {
        int(seed)
        for key in ("selection_seed_pairs", "audit_seed_pairs")
        for pair in parent_summary[key]
        for seed in pair
    }
    if forbidden_seeds & (set(selection_seeds) | set(audit_seeds)):
        raise ValueError("risk replay seeds overlap parent first/second-pass seeds")
    paths = sorted(parent_root.glob("episode_*/*.npz"))
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise FileNotFoundError(parent_root)
    device = torch.device(args.device)
    models, normalization, feedback_mean, feedback_std, checkpoints = load_critic_ensemble(
        critic_dir, device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    selection_values, audit_values = [], []
    try:
        for index, parent_path in enumerate(paths, 1):
            arrays, metadata = process_one(
                parent_path, source_root, config, args, radii,
                selection_seeds, audit_seeds, models, normalization,
                feedback_mean, feedback_std, device,
            )
            episode_dir = staging / parent_path.parent.name
            episode_dir.mkdir(exist_ok=True)
            target = episode_dir / parent_path.name
            np.savez_compressed(target, **arrays)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            selection_values.append(arrays["proposal_output_cost_by_seed"])
            audit_values.append(arrays["audit_proposal_output_cost_by_seed"])
            if index % 20 == 0 or index == len(paths):
                print(f"[{index:04d}/{len(paths):04d}] risk replay ({time.perf_counter()-started:.1f}s)", flush=True)
        shutil.copy2(parent_root / "splits.json", staging / "splits.json")
        shutil.copy2(parent_root / "teacher_config.json", staging / "teacher_config.json")
        selection = np.asarray(selection_values, np.float32)
        audit = np.asarray(audit_values, np.float32)
        actual_center_count = 1 + 2 * len(radii)
        total_candidates = (
            len(paths) * 4 * actual_center_count * len(selection_seeds)
            * args.samples_per_center_seed
        )
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root),
            "parent_labels": str(parent_root),
            "critic_dir": str(critic_dir),
            "critic_checkpoints": checkpoints,
            "critic_checkpoint_sha256": [sha256_file(Path(value)) for value in checkpoints],
            "snapshot_count": len(paths),
            "repeats_per_partition": int(selection.shape[1]),
            "radii_sigma": radii,
            "center_names": center_names(radii),
            "center_count": actual_center_count,
            "selection_seeds": selection_seeds,
            "audit_seeds": audit_seeds,
            "samples_per_center_seed": args.samples_per_center_seed,
            "second_noise_scale": args.second_noise_scale,
            "candidate_rollout_count": total_candidates,
            "selection": aggregate_summary(selection),
            "audit": aggregate_summary(audit),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "semantics": (
                "No new states or first-pass rollouts. Repeated second-pass rewards "
                "are evaluated along the frozen feedback critic direction with common "
                "random numbers. No analytic DBM gradient is used."
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
