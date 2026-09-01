#!/usr/bin/env python3
"""Generate an isolated train-only 40--100-kph fixed-DBM collection plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_TRACK = Path("car_planner/assets/cuc_inside.csv")
DEFAULT_OUTPUT = Path(
    "scripts/model_verify/fixed_dbm_highspeed_train_20260828_v2.json"
)
DEFAULT_COLLECTION = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_highspeed_train_20260828_v2"
)
SPEEDS_KPH = (40.0, 55.0, 70.0, 85.0, 100.0)
REGIMES = (
    "steady",
    "underspeed_recovery",
    "overspeed_recovery",
    "lateral_recovery",
    "heading_recovery",
    "combined_recovery",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--mppi-seed-start", type=int, default=8800)
    parser.add_argument("--episodes-per-stratum", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-snapshots", type=int, default=20)
    return parser.parse_args()


def signed_uniform(
    rng: np.random.Generator, minimum: float, maximum: float
) -> float:
    return (-1.0 if rng.random() < 0.5 else 1.0) * float(
        rng.uniform(minimum, maximum)
    )


def scenario_values(
    regime: str, speed: float, rng: np.random.Generator
) -> tuple[float, float, float, float, float]:
    # These are state-distribution definitions, not changes to DBM parameters.
    # vx stays in the target-speed neighborhood so the train-only collection
    # actually covers 40--100 kph rather than mostly acceleration transients.
    if regime == "steady":
        lateral = float(rng.uniform(-0.035, 0.035))
        heading = float(rng.uniform(-0.035, 0.035))
        vx = speed * float(rng.uniform(0.92, 1.05))
        vy = float(rng.uniform(-0.02, 0.02))
        yawrate = float(rng.uniform(-0.08, 0.08))
    elif regime == "underspeed_recovery":
        lateral = float(rng.uniform(-0.10, 0.10))
        heading = float(rng.uniform(-0.08, 0.08))
        vx = speed * float(rng.uniform(0.70, 0.88))
        vy = float(rng.uniform(-0.04, 0.04))
        yawrate = float(rng.uniform(-0.16, 0.16))
    elif regime == "overspeed_recovery":
        lateral = float(rng.uniform(-0.10, 0.10))
        heading = float(rng.uniform(-0.08, 0.08))
        vx = speed * float(rng.uniform(1.04, 1.12))
        vy = float(rng.uniform(-0.04, 0.04))
        yawrate = float(rng.uniform(-0.16, 0.16))
    elif regime == "lateral_recovery":
        lateral = signed_uniform(rng, 0.12, 0.26)
        heading = float(rng.uniform(-0.13, 0.13))
        vx = speed * float(rng.uniform(0.82, 1.05))
        vy = signed_uniform(rng, 0.02, 0.08)
        yawrate = float(rng.uniform(-0.30, 0.30))
    elif regime == "heading_recovery":
        lateral = float(rng.uniform(-0.16, 0.16))
        heading = signed_uniform(rng, 0.12, 0.24)
        vx = speed * float(rng.uniform(0.82, 1.08))
        vy = float(rng.uniform(-0.07, 0.07))
        yawrate = signed_uniform(rng, 0.10, 0.38)
    elif regime == "combined_recovery":
        lateral = signed_uniform(rng, 0.17, 0.27)
        heading = signed_uniform(rng, 0.14, 0.24)
        vx = speed * float(rng.uniform(0.75, 1.08))
        vy = signed_uniform(rng, 0.04, 0.10)
        yawrate = signed_uniform(rng, 0.20, 0.45)
    else:
        raise ValueError(f"unknown regime: {regime}")
    return lateral, heading, vx, vy, yawrate


def wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen plan: {args.output}")
    if args.episodes_per_stratum < 1:
        raise ValueError("--episodes-per-stratum must be positive")
    if args.num_samples < 4 or args.num_samples % 2:
        raise ValueError("--num-samples must be an even integer >= 4")
    if args.start_step < 0:
        raise ValueError("--start-step cannot be negative")
    for name in ("stride", "max_snapshots"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    track = np.loadtxt(args.track, delimiter=",", skiprows=1, dtype=np.float64)
    if track.ndim != 2 or track.shape[1] < 4:
        raise ValueError(f"unexpected track shape: {track.shape}")
    rng = np.random.default_rng(args.seed)
    specifications = [
        (speed_kph, regime)
        for speed_kph in SPEEDS_KPH
        for regime in REGIMES
        for _ in range(args.episodes_per_stratum)
    ]
    edges = np.linspace(0, len(track), len(specifications) + 1, dtype=np.int64)
    track_indices = np.asarray(
        [rng.integers(edges[i], edges[i + 1]) for i in range(len(specifications))],
        dtype=np.int64,
    )
    rng.shuffle(track_indices)

    episodes = []
    for index, ((speed_kph, regime), track_index) in enumerate(
        zip(specifications, track_indices)
    ):
        speed = speed_kph / 3.6
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
                "split": "train",
                "scenario_class": regime,
                "speed_kph": speed_kph,
                "track_index": int(track_index),
                "lateral_offset_m": lateral,
                "heading_error_rad": heading_error,
                "reference_speed_mps": speed,
                "initial_state": (
                    f"{x:.6f},{y:.6f},{yaw:.6f},{vx:.6f},{vy:.6f},"
                    f"{yawrate:.6f}"
                ),
                "mppi_seed": args.mppi_seed_start + index,
            }
        )

    maximum_speed = max(SPEEDS_KPH) / 3.6
    plan = {
        "format_version": 1,
        "description": (
            "Isolated train-only fixed-DBM high-speed initial-state collection. "
            "Five target speeds from 40 to 100 kph and six state regimes; "
            "snapshots start immediately so the recorded physical state remains "
            "inside the requested speed domain. Never merge into the historical "
            "1.2--2.8-m/s train/validation/test splits."
        ),
        "generation_seed": args.seed,
        "source_track": str(args.track.resolve()),
        "collection_dir": str(args.collection_dir.resolve()),
        "speed_domain": {
            "units": "kph",
            "values": list(SPEEDS_KPH),
            "mps_values": [value / 3.6 for value in SPEEDS_KPH],
            "isolated_from_low_speed_splits": True,
        },
        "collection": {
            "start_step": args.start_step,
            "stride": args.stride,
            "max_snapshots": args.max_snapshots,
            "trace_start_step": 0,
            "num_samples": args.num_samples,
            "num_iterations": 1,
            "reference_speed_max_mps": max(30.0, maximum_speed),
        },
        "split_counts": {
            "train": len(episodes),
            "validation": 0,
            "test": 0,
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "collection_dir": str(args.collection_dir),
                "episode_count": len(episodes),
                "snapshot_count": len(episodes) * args.max_snapshots,
                "speed_kph": list(SPEEDS_KPH),
                "scenario_classes": list(REGIMES),
                "split_counts": plan["split_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
