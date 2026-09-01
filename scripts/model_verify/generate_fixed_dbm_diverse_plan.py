#!/usr/bin/env python3
"""Generate a deterministic, episode-diverse fixed-DBM collection plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_TRACK = Path("car_planner/assets/cuc_inside.csv")
DEFAULT_OUTPUT = Path(
    "scripts/model_verify/fixed_dbm_policy_diverse_20260805_v1.json"
)
DEFAULT_COLLECTION = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
SPEEDS = (1.2, 1.6, 2.0, 2.4, 2.8)
TRAIN_REGIMES = (
    "steady",
    "cold_start",
    "lateral_recovery",
    "heading_recovery",
    "dynamic_recovery",
    "combined_recovery",
)
HELDOUT_REGIMES = ("heldout_nominal", "heldout_mixed", "heldout_recovery")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--mppi-seed-start", type=int, default=5600)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--train-episodes-per-stratum",
        type=int,
        default=3,
        help="Independent episodes for each of the 5 speed x 6 train regimes.",
    )
    parser.add_argument(
        "--heldout-episodes-per-stratum",
        type=int,
        default=1,
        help="Episodes for each speed x heldout regime in validation and test.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Generate an additive train-only plan without validation/test episodes.",
    )
    parser.add_argument("--episode-index-start", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=250)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-snapshots", type=int, default=20)
    return parser.parse_args()


def signed_uniform(
    rng: np.random.Generator, minimum: float, maximum: float
) -> float:
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * float(rng.uniform(minimum, maximum))


def scenario_values(
    regime: str, speed: float, rng: np.random.Generator
) -> tuple[float, float, float, float, float]:
    if regime in ("steady", "heldout_nominal"):
        lateral = float(rng.uniform(-0.035, 0.035))
        heading = float(rng.uniform(-0.035, 0.035))
        vx = speed * float(rng.uniform(0.85, 1.10))
        vy = float(rng.uniform(-0.015, 0.015))
        yawrate = float(rng.uniform(-0.06, 0.06))
    elif regime == "cold_start":
        lateral = float(rng.uniform(-0.12, 0.12))
        heading = float(rng.uniform(-0.10, 0.10))
        vx = speed * float(rng.uniform(0.0, 0.25))
        vy = float(rng.uniform(-0.025, 0.025))
        yawrate = float(rng.uniform(-0.12, 0.12))
    elif regime == "lateral_recovery":
        lateral = signed_uniform(rng, 0.12, 0.26)
        heading = float(rng.uniform(-0.13, 0.13))
        vx = speed * float(rng.uniform(0.40, 1.10))
        vy = signed_uniform(rng, 0.015, 0.065)
        yawrate = float(rng.uniform(-0.28, 0.28))
    elif regime == "heading_recovery":
        lateral = float(rng.uniform(-0.16, 0.16))
        heading = signed_uniform(rng, 0.12, 0.24)
        vx = speed * float(rng.uniform(0.40, 1.15))
        vy = float(rng.uniform(-0.055, 0.055))
        yawrate = signed_uniform(rng, 0.10, 0.34)
    elif regime in ("dynamic_recovery", "heldout_mixed"):
        lateral = float(rng.uniform(-0.20, 0.20))
        heading = float(rng.uniform(-0.19, 0.19))
        vx = speed * float(rng.uniform(0.25, 1.20))
        vy = signed_uniform(rng, 0.025, 0.075)
        yawrate = signed_uniform(rng, 0.16, 0.38)
    elif regime in ("combined_recovery", "heldout_recovery"):
        lateral = signed_uniform(rng, 0.17, 0.27)
        heading = signed_uniform(rng, 0.14, 0.24)
        vx = speed * float(rng.uniform(0.15, 1.05))
        vy = signed_uniform(rng, 0.035, 0.080)
        yawrate = signed_uniform(rng, 0.20, 0.40)
    else:
        raise ValueError(f"unknown scenario regime: {regime}")
    return lateral, heading, vx, vy, yawrate


def wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def main() -> None:
    args = parse_args()
    if args.num_samples < 4 or args.num_samples % 2:
        raise ValueError("--num-samples must be an even integer >= 4")
    for name in (
        "train_episodes_per_stratum",
        "heldout_episodes_per_stratum",
        "start_step",
        "stride",
        "max_snapshots",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.episode_index_start < 0:
        raise ValueError("--episode-index-start must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen plan: {args.output}")
    track = np.loadtxt(args.track, delimiter=",", skiprows=1, dtype=np.float64)
    if track.ndim != 2 or track.shape[1] < 4:
        raise ValueError(f"unexpected track shape: {track.shape}")
    rng = np.random.default_rng(args.seed)
    specifications: list[tuple[str, str, float]] = []
    for speed in SPEEDS:
        for regime in TRAIN_REGIMES:
            specifications.extend(
                ("train", regime, speed)
                for _ in range(args.train_episodes_per_stratum)
            )
    if not args.train_only:
        for split in ("validation", "test"):
            for speed in SPEEDS:
                for regime in HELDOUT_REGIMES:
                    specifications.extend(
                        (split, regime, speed)
                        for _ in range(args.heldout_episodes_per_stratum)
                    )
    total = len(specifications)
    # One index from every equal-length track-phase bin gives full-lap coverage
    # without reusing the exact same initial phase in another episode.
    edges = np.linspace(0, len(track), total + 1, dtype=np.int64)
    track_indices = np.asarray(
        [rng.integers(edges[i], edges[i + 1]) for i in range(total)],
        dtype=np.int64,
    )
    rng.shuffle(track_indices)
    episodes = []
    for relative_index, ((split, regime, speed), track_index) in enumerate(
        zip(specifications, track_indices)
    ):
        index = args.episode_index_start + relative_index
        lateral, heading_error, vx, vy, yawrate = scenario_values(
            regime, speed, rng
        )
        base_x = float(track[track_index, 0])
        base_y = float(track[track_index, 1])
        base_yaw = float(track[track_index, 3])
        x = base_x - np.sin(base_yaw) * lateral
        y = base_y + np.cos(base_yaw) * lateral
        yaw = wrap_angle(base_yaw + heading_error)
        episodes.append(
            {
                "episode_id": f"episode_{index:03d}",
                "split": split,
                "scenario_class": regime,
                "track_index": int(track_index),
                "lateral_offset_m": lateral,
                "heading_error_rad": heading_error,
                "reference_speed_mps": speed,
                "initial_state": (
                    f"{x:.6f},{y:.6f},{yaw:.6f},{vx:.6f},{vy:.6f},"
                    f"{yawrate:.6f}"
                ),
                "mppi_seed": args.mppi_seed_start + relative_index,
            }
        )
    split_counts = {
        split: sum(episode["split"] == split for episode in episodes)
        for split in ("train", "validation", "test")
    }
    plan = {
        "format_version": 1,
        "description": (
            "Diverse fixed-DBM proposal-policy collection: five speeds, "
            "continuous track/error/dynamic-state coverage, more independent "
            "episodes, and a reduced per-step MPPI candidate budget."
        ),
        "generation_seed": args.seed,
        "source_track": str(args.track.resolve()),
        "collection_dir": str(args.collection_dir.resolve()),
        "collection": {
            "start_step": args.start_step,
            "stride": args.stride,
            "max_snapshots": args.max_snapshots,
            "trace_start_step": 0,
            "num_samples": args.num_samples,
            "num_iterations": 1,
        },
        "split_counts": split_counts,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "episode_count": total,
                "snapshot_count": total * plan["collection"]["max_snapshots"],
                "split_counts": plan["split_counts"],
                "num_samples": args.num_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
