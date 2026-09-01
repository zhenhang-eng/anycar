#!/usr/bin/env python3
"""Summarize an immutable fixed-DBM high-speed snapshot collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_COLLECTION = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_highspeed_train_20260828_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_collection_summary_20260828_v1/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary input must be nonempty and finite")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    collection = args.collection.resolve()
    plan_path = collection / "scenario_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_rows = {row["episode_id"]: row for row in plan["episodes"]}
    expected_per_episode = int(plan["collection"]["max_snapshots"])
    expected_snapshot_count = len(plan_rows) * expected_per_episode
    candidate_count = int(plan["collection"]["num_samples"])
    episode_dirs = sorted(collection.glob("episode_*"))
    if len(episode_dirs) != len(plan_rows):
        raise AssertionError(
            f"episode count mismatch: {len(episode_dirs)} != {len(plan_rows)}"
        )

    values: dict[str, list[float]] = defaultdict(list)
    by_speed: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenarios: dict[str, int] = defaultdict(int)
    trace_rows = 0
    snapshot_count = 0
    repository_hashes: set[str] = set()
    dbm_params: set[str] = set()
    cost_weights: set[str] = set()

    for episode_dir in episode_dirs:
        episode_id = episode_dir.name
        row = plan_rows[episode_id]
        speed_key = f"{float(row['speed_kph']):g}"
        scenarios[str(row["scenario_class"])] += 1
        manifest = json.loads((episode_dir / "manifest.json").read_text())
        if int(manifest["snapshot_count"]) != expected_per_episode:
            raise AssertionError(
                f"{episode_id}: expected {expected_per_episode} snapshots"
            )
        if manifest["controller_backend"] != "dbm":
            raise AssertionError(f"{episode_id}: controller is not DBM")
        repository_hashes.add(json.dumps(manifest["repository"], sort_keys=True))
        dbm_params.add(json.dumps(manifest["rollout_model"]["dbm_params"], sort_keys=True))
        cost_weights.add(json.dumps(manifest["cost_weights_at_collection"], sort_keys=True))
        trace_rows += sum(1 for _ in (episode_dir / "closed_loop_trace.jsonl").open())

        for snapshot_path in sorted((episode_dir / "snapshots").glob("step_*.npz")):
            with np.load(snapshot_path, allow_pickle=False) as data:
                state = np.asarray(data["initial_state_six"], np.float64)
                cost = np.asarray(data["cost"], np.float64)
                clipped = np.asarray(data["sampled_knots_clipped"], np.bool_)
                reference_speed = float(data["reference_speed_override_mps"])
                frenet = np.asarray(data["frenet_pose"], np.float64)
                if not (
                    np.all(np.isfinite(state))
                    and np.all(np.isfinite(cost))
                    and np.isfinite(reference_speed)
                    and np.all(np.isfinite(frenet))
                ):
                    raise AssertionError(f"{snapshot_path}: nonfinite values")
                row_values = {
                    "reference_speed_mps": reference_speed,
                    "observed_vx_mps": float(state[3]),
                    "observed_vy_mps": float(state[4]),
                    "observed_yawrate_rps": float(state[5]),
                    "frenet_lateral_m": float(frenet[1]),
                    "frenet_heading_error_rad": float(frenet[2]),
                    "best_candidate_cost": float(np.min(cost)),
                    "median_candidate_cost": float(np.median(cost)),
                    "knot_clip_fraction": float(np.mean(clipped)),
                }
                for name, value in row_values.items():
                    values[name].append(value)
                    by_speed[speed_key][name].append(value)
            snapshot_count += 1

    if snapshot_count != expected_snapshot_count:
        raise AssertionError(
            f"expected {expected_snapshot_count} snapshots, got {snapshot_count}"
        )
    if len(repository_hashes) != 1 or len(dbm_params) != 1 or len(cost_weights) != 1:
        raise AssertionError("collection contract differs across episodes")

    target_lower = min(plan["speed_domain"]["mps_values"])
    target_upper = max(plan["speed_domain"]["mps_values"])
    observed_vx = np.asarray(values["observed_vx_mps"], np.float64)
    target_fraction = float(np.mean(
        (observed_vx >= target_lower) & (observed_vx <= target_upper)
    ))
    qualification = (
        "HIGHSPEED_TRAIN_COLLECTION_COMPLETE_VALIDATED"
        if target_fraction >= 0.75
        else (
            "HIGH_REFERENCE_SPEED_STRESS_COLLECTION_COMPLETE_VALIDATED_"
            "ACTUAL_SPEED_TARGET_MISSED"
        )
    )
    summary = {
        "qualification": qualification,
        "collection": str(collection),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "track_sha256": sha256(collection / "episode_000" / "track.npz"),
        "episode_count": len(episode_dirs),
        "snapshot_count": snapshot_count,
        "candidate_rollout_count": snapshot_count * candidate_count,
        "closed_loop_trace_row_count": trace_rows,
        "split_counts": plan["split_counts"],
        "speed_domain": plan["speed_domain"],
        "actual_40_100_kph_fraction": target_fraction,
        "scenario_episode_counts": dict(sorted(scenarios.items())),
        "overall": {name: metrics(data) for name, data in sorted(values.items())},
        "by_speed_kph": {
            speed: {name: metrics(data) for name, data in sorted(group.items())}
            for speed, group in sorted(by_speed.items(), key=lambda item: float(item[0]))
        },
        "contract": {
            "repository": json.loads(next(iter(repository_hashes))),
            "dbm_params": json.loads(next(iter(dbm_params))),
            "cost_weights": json.loads(next(iter(cost_weights))),
            "formal_validation_or_test_created": False,
            "isolated_from_historical_low_speed_splits": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
