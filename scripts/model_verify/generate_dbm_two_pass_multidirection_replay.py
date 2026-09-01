#!/usr/bin/env python3
"""Generate repeated DBM rewards for a feedback-derived direction bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from generate_dbm_multicenter_teacher import load_config, make_controller
from generate_dbm_proposal_teacher import repository_state, sha256_file
from generate_dbm_two_pass_risk_replay_labels import evaluate_center_bank


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-two-pass-multidirection-replay-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DIRECTION_NAMES = (
    "critic",
    "first_best",
    "soft_shift",
    "negative_gradient",
    "preconditioned_gradient",
    "critic_plus_best",
    "critic_plus_preconditioned",
    "best_plus_preconditioned",
)
DEFAULT_RADII = (0.03, 0.06, 0.10, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radii", default=",".join(map(str, DEFAULT_RADII)))
    parser.add_argument("--selection-seeds", default="28001,28002,28003,28004")
    parser.add_argument("--audit-seeds", default="28101,28102,28103,28104")
    parser.add_argument("--samples-per-center-seed", type=int, default=64)
    parser.add_argument("--second-noise-scale", type=float, default=0.10)
    parser.add_argument("--preconditioner-damping", type=float, default=0.10)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_numbers(text: str, value_type: type) -> list[Any]:
    values = [value_type(value) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("values must be non-empty and distinct")
    return values


def normalize_direction(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32).reshape(8, 2)
    maximum = float(np.max(np.abs(value)))
    if not np.isfinite(maximum) or maximum < 1e-6:
        return np.asarray(fallback, np.float32).reshape(8, 2)
    return (value / maximum).astype(np.float32)


def direction_bank(
    feedback: np.ndarray,
    first_knots: np.ndarray,
    first_cost: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    critic_direction: np.ndarray,
    damping: float,
) -> np.ndarray:
    banks = []
    for one_feedback, knots, costs, one_critic in zip(
        feedback, first_knots, first_cost, critic_direction
    ):
        critic = normalize_direction(one_critic, np.ones((8, 2), np.float32))
        best = (knots[int(np.argmin(costs))] - base) / sigma.reshape(1, 2)
        best = normalize_direction(best, critic)
        soft = normalize_direction(one_feedback[48:64], critic)
        gradient = one_feedback[16:32]
        hessian = np.maximum(one_feedback[32:48], 0.0)
        negative = normalize_direction(-gradient, critic)
        preconditioned = normalize_direction(
            -gradient / (hessian + float(damping)), critic
        )
        values = (
            critic,
            best,
            soft,
            negative,
            preconditioned,
            normalize_direction(critic + best, critic),
            normalize_direction(critic + preconditioned, critic),
            normalize_direction(best + preconditioned, critic),
        )
        banks.append(values)
    return np.asarray(banks, np.float32)


def action_names(radii: list[float]) -> tuple[str, ...]:
    names = ["guided_anchor"]
    for direction in DIRECTION_NAMES:
        for radius in radii:
            names.append(
                f"{direction}_pos_{radius:.3f}".replace(".", "p")
            )
    return tuple(names)


def make_centers(
    anchors: np.ndarray,
    directions: np.ndarray,
    radii: list[float],
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    banks = []
    for anchor, direction_bank_value in zip(anchors, directions):
        centers = [anchor]
        for direction in direction_bank_value:
            for radius in radii:
                centers.append(
                    anchor + float(radius) * direction * sigma.reshape(1, 2)
                )
        banks.append(centers)
    raw = np.asarray(banks, np.float32)
    return raw, np.clip(raw, action_min, action_max).astype(np.float32)


def process_one(
    parent_path: Path,
    source_root: Path,
    risk_root: Path,
    config: dict[str, Any],
    radii: list[float],
    selection_seeds: list[int],
    audit_seeds: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episode = parent_path.parent.name
    source_path = source_root / episode / "snapshots" / parent_path.name
    risk_path = risk_root / episode / parent_path.name
    source_hash = sha256_file(source_path)
    parent_hash = sha256_file(parent_path)
    risk_hash = sha256_file(risk_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(
        parent_path, allow_pickle=False
    ) as parent, np.load(risk_path, allow_pickle=False) as risk:
        if str(parent["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{parent_path}: source hash mismatch")
        if str(risk["parent_label_sha256"]) != parent_hash:
            raise AssertionError(f"{risk_path}: parent hash mismatch")
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], np.float32)
        action_min = np.asarray(params["action_min"], np.float32)
        action_max = np.asarray(params["action_max"], np.float32)
        controller, backend = make_controller(source, config, device)
        common = dict(
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
            "risk_label_sha256": np.asarray(risk_hash),
            "action_names": np.asarray(action_names(radii)),
            "direction_names": np.asarray(DIRECTION_NAMES),
            "radii_sigma": np.asarray(radii, np.float32),
        }
        base = np.asarray(parent["base_center_knots"], np.float32)
        for prefix, seeds in (("", selection_seeds), ("audit_", audit_seeds)):
            parent_prefix = "" if not prefix else "audit_"
            anchors = np.asarray(parent[f"{parent_prefix}guided_center_knots"], np.float32)
            feedback = np.asarray(parent[f"{parent_prefix}first_pass_feedback"], np.float32)
            critic = np.asarray(
                risk[f"{prefix}normalized_critic_direction"], np.float32
            )
            directions = direction_bank(
                feedback,
                np.asarray(parent[f"{parent_prefix}first_pass_knots"], np.float32),
                np.asarray(parent[f"{parent_prefix}first_pass_cost"], np.float32),
                base,
                sigma,
                critic,
                args.preconditioner_damping,
            )
            raw, centers = make_centers(
                anchors, directions, radii, sigma, action_min, action_max
            )
            result = evaluate_center_bank(centers=centers, seeds=seeds, **common)
            arrays.update({
                f"{prefix}first_pass_seed": np.asarray(
                    parent[f"{parent_prefix}first_pass_seed"], np.int64
                ),
                f"{prefix}feedback_sha256": np.asarray([
                    hashlib.sha256(value.tobytes()).hexdigest() for value in feedback
                ]),
                f"{prefix}guided_center_knots": anchors,
                f"{prefix}directions": directions,
                f"{prefix}raw_centers": raw,
                f"{prefix}centers": centers,
                f"{prefix}evaluation_seeds": np.asarray(seeds, np.int64),
                **{f"{prefix}{name}": value for name, value in result.items()},
            })
    return arrays, {
        "format_version": FORMAT_VERSION,
        "generator_id": GENERATOR_ID,
        "episode_id": episode,
        "control_step": int(parent_path.stem.removeprefix("step_")),
        "source_snapshot_sha256": source_hash,
        "parent_label_sha256": parent_hash,
        "risk_label_sha256": risk_hash,
    }


def aggregate(costs: np.ndarray) -> dict[str, Any]:
    advantage = costs[:, :, 0:1] - costs
    context = advantage.mean(axis=3)
    return {
        "mean_advantage": context.mean(axis=(0, 1)).tolist(),
        "median_advantage": np.median(context, axis=(0, 1)).tolist(),
        "positive_context_fraction": np.mean(context > 0.0, axis=(0, 1)).tolist(),
        "p10_seed_advantage": np.quantile(
            advantage, 0.10, axis=(0, 1, 3)
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    radii = parse_numbers(args.radii, float)
    if radii != sorted(radii) or radii[0] <= 0 or radii[-1] > 0.25:
        raise ValueError("radii must be sorted in (0,0.25]")
    selection_seeds = parse_numbers(args.selection_seeds, int)
    audit_seeds = parse_numbers(args.audit_seeds, int)
    if set(selection_seeds) & set(audit_seeds):
        raise ValueError("selection/audit seeds overlap")
    source_root = args.source.resolve()
    parent_root = args.parent_labels.resolve()
    risk_root = args.risk_labels.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    risk_summary = json.loads((risk_root / "summary.json").read_text())
    forbidden = {
        int(seed)
        for key in ("selection_seed_pairs", "audit_seed_pairs")
        for pair in parent_summary[key]
        for seed in pair
    } | set(risk_summary["selection_seeds"]) | set(risk_summary["audit_seeds"])
    if forbidden & (set(selection_seeds) | set(audit_seeds)):
        raise ValueError("new reward seeds overlap parent/replay seeds")
    config = load_config(parent_root / "teacher_config.json")
    paths = sorted(parent_root.glob("episode_*/*.npz"))
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise FileNotFoundError(parent_root)
    device = torch.device(args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    selection_cost, audit_cost = [], []
    try:
        for index, path in enumerate(paths, 1):
            arrays, metadata = process_one(
                path, source_root, risk_root, config, radii,
                selection_seeds, audit_seeds, args, device,
            )
            target_dir = staging / path.parent.name
            target_dir.mkdir(exist_ok=True)
            target = target_dir / path.name
            np.savez_compressed(target, **arrays)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            selection_cost.append(arrays["proposal_output_cost_by_seed"])
            audit_cost.append(arrays["audit_proposal_output_cost_by_seed"])
            if index % 20 == 0 or index == len(paths):
                print(
                    f"[{index:04d}/{len(paths):04d}] multidirection replay "
                    f"({time.perf_counter()-started:.1f}s)", flush=True
                )
        shutil.copy2(parent_root / "splits.json", staging / "splits.json")
        shutil.copy2(parent_root / "teacher_config.json", staging / "teacher_config.json")
        selection = np.asarray(selection_cost, np.float32)
        audit = np.asarray(audit_cost, np.float32)
        center_count = 1 + len(DIRECTION_NAMES) * len(radii)
        rollout_count = (
            len(paths) * 4 * center_count * len(selection_seeds)
            * args.samples_per_center_seed
        )
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root),
            "parent_labels": str(parent_root),
            "risk_labels": str(risk_root),
            "snapshot_count": len(paths),
            "repeats_per_partition": int(selection.shape[1]),
            "direction_names": list(DIRECTION_NAMES),
            "radii_sigma": radii,
            "action_names": list(action_names(radii)),
            "center_count": center_count,
            "selection_seeds": selection_seeds,
            "audit_seeds": audit_seeds,
            "samples_per_center_seed": args.samples_per_center_seed,
            "second_noise_scale": args.second_noise_scale,
            "preconditioner_damping": args.preconditioner_damping,
            "candidate_rollout_count": rollout_count,
            "selection": aggregate(selection),
            "audit": aggregate(audit),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path.cwd()),
            "semantics": (
                "Eight directions are reconstructed only from first-pass feedback and "
                "the frozen feedback critic. Teacher centers are not used. All rewards "
                "come from forward fixed-DBM second-pass rollouts."
            ),
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "ok", "output": str(output), "snapshot_count": len(paths),
        "candidate_rollout_count": rollout_count,
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
