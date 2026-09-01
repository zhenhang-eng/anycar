#!/usr/bin/env python3
"""Sequentially collect and validate fixed-DBM episodes from a frozen plan."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PLAN = Path(__file__).with_name(
    "fixed_dbm_expansion_pilot_20260804_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--episodes",
        default="",
        help="Optional comma-separated episode IDs; empty runs the whole plan.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already complete episodes instead of refusing them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("format_version") != 1:
        raise ValueError("scenario plan format_version must be 1")
    if plan.get("status") == "rejected":
        raise ValueError(f"scenario plan is rejected: {plan.get('rejection_reason')}")
    collection = plan.get("collection", {})
    if int(collection.get("start_step", -1)) < 0:
        raise ValueError("collection.start_step cannot be negative")
    for name in ("stride", "max_snapshots", "num_samples"):
        if int(collection.get(name, 0)) <= 0:
            raise ValueError(f"collection.{name} must be positive")
    episodes = plan.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("scenario plan must contain episodes")
    ids = [episode.get("episode_id") for episode in episodes]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("episode IDs must be unique and nonempty")
    seeds = [int(episode["mppi_seed"]) for episode in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError("MPPI seeds must be unique")
    reference_speed_max = float(
        collection.get("reference_speed_max_mps", 10.0)
    )
    if not 0.0 < reference_speed_max <= 100.0:
        raise ValueError(
            "collection.reference_speed_max_mps must be in (0, 100]"
        )
    for episode in episodes:
        state = [float(value) for value in episode["initial_state"].split(",")]
        if len(state) != 6:
            raise ValueError(f"{episode['episode_id']}: initial_state must have 6 values")
        speed = float(episode["reference_speed_mps"])
        if not 0.0 <= speed <= reference_speed_max:
            raise ValueError(f"{episode['episode_id']}: invalid reference speed")
    if "split_counts" in plan:
        split_counts = plan["split_counts"]
        expected_order = []
        for split in ("train", "validation", "test"):
            expected_order.extend([split] * int(split_counts[split]))
        actual_order = [episode.get("split") for episode in episodes]
        if actual_order != expected_order:
            raise ValueError(
                "episodes must be ordered train/validation/test to match T0 split counts"
            )


def validate_episode(repository: Path, episode_dir: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            str(repository / "scripts/model_verify/validate_mppi_closed_loop_dataset.py"),
            str(episode_dir),
        ),
        cwd=repository,
        check=True,
    )


def run_episode(
    repository: Path,
    collection_dir: Path,
    collection: dict[str, Any],
    episode: dict[str, Any],
    dry_run: bool,
) -> None:
    command = [
        "ros2",
        "launch",
        "car_ros2",
        "car_sim.launch.py",
        "mppi_backend:=dbm",
        f"mppi_seed:={int(episode['mppi_seed'])}",
        f"mppi_num_samples:={int(collection['num_samples'])}",
        f"mppi_reference_speed:={float(episode['reference_speed_mps'])}",
        "mppi_reference_speed_max:="
        f"{float(collection.get('reference_speed_max_mps', 10.0))}",
        f"mppi_dataset_dir:={collection_dir}",
        f"mppi_dataset_episode_id:={episode['episode_id']}",
        f"mppi_dataset_start_step:={int(collection['start_step'])}",
        f"mppi_dataset_stride:={int(collection['stride'])}",
        f"mppi_dataset_max_snapshots:={int(collection['max_snapshots'])}",
        "mppi_dataset_shutdown_on_complete:=True",
        f"sim_initial_state:={episode['initial_state']}",
    ]
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log_dir = collection_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{episode['episode_id']}.log"
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    environment.setdefault("ROS_LOG_DIR", "/tmp/anycar_ros_logs")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repository,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture ROS launch output")
        for line in process.stdout:
            log.write(line)
            if "MPPI snapshot written" in line or "process has finished cleanly" in line:
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    collection_dir = Path(plan["collection_dir"]).resolve()
    requested = {value.strip() for value in args.episodes.split(",") if value.strip()}
    known = {episode["episode_id"] for episode in plan["episodes"]}
    if requested - known:
        raise ValueError(f"unknown episodes: {sorted(requested - known)}")
    episodes = [
        episode
        for episode in plan["episodes"]
        if not requested or episode["episode_id"] in requested
    ]
    if not args.dry_run:
        collection_dir.mkdir(parents=True, exist_ok=True)
        frozen_plan = collection_dir / "scenario_plan.json"
        if frozen_plan.exists():
            if json.loads(frozen_plan.read_text()) != plan:
                raise ValueError(f"existing frozen plan differs: {frozen_plan}")
        else:
            shutil.copy2(plan_path, frozen_plan)
    completed = []
    for episode in episodes:
        episode_dir = collection_dir / episode["episode_id"]
        if episode_dir.exists() and any(episode_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(
                    f"episode already exists; use --resume after inspection: {episode_dir}"
                )
            validate_episode(repository, episode_dir)
            print(f"validated existing {episode['episode_id']}", flush=True)
            completed.append(episode["episode_id"])
            continue
        run_episode(repository, collection_dir, plan["collection"], episode, args.dry_run)
        if not args.dry_run:
            validate_episode(repository, episode_dir)
            completed.append(episode["episode_id"])
    print(
        json.dumps(
            {
                "status": "dry-run" if args.dry_run else "ok",
                "collection_dir": str(collection_dir),
                "episodes": completed if not args.dry_run else [e["episode_id"] for e in episodes],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
