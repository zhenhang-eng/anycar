#!/usr/bin/env python3
"""Independently validate key hard-guard closed-loop pilot metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RUN = Path("outputs/mppi_proposal/hard_guard_closed_loop_pilot_20260813_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def realized(values: list[dict], start: int) -> tuple[float, float, float, float]:
    values = [row for row in values if row["control_step"] >= start]
    state = np.asarray([row["state"] for row in values], np.float64)
    current = np.asarray([row["current_action"] for row in values], np.float64)
    action = np.asarray([row["executed_action"] for row in values], np.float64)
    frenet = np.asarray([row["frenet_pose"] for row in values], np.float64)
    speed = np.asarray([row["reference"][0][3] for row in values], np.float64)
    rate = action - current
    stage = (
        5 * frenet[:, 1] ** 2 + 10 * frenet[:, 2] ** 2
        + (state[:, 3] - speed) ** 2
        + 0.1 * rate[:, 0] ** 2 + 0.1 * rate[:, 1] ** 2
    )
    return (
        float(stage.sum()),
        float(np.sqrt(np.mean(frenet[:, 1] ** 2))),
        float(np.sqrt(np.mean(frenet[:, 2] ** 2))),
        float(np.sqrt(np.mean((state[:, 3] - speed) ** 2))),
    )


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    root = Path(summary["root"])
    baseline = rows(root / summary["baseline_episode"] / "closed_loop_trace.jsonl")
    guard = rows(root / summary["guard_episode"] / "closed_loop_trace.jsonl")
    if len(baseline) != 301 or len(guard) != 301:
        raise AssertionError("unexpected trace lengths")

    errors = {}
    for window, start in (("all_steps", 0), ("history_complete", 250)):
        for arm, values in (("baseline", baseline), ("guard", guard)):
            actual = realized(values, start)
            stored = summary["windows"][window][arm]
            expected = (
                stored["realized_stage_cost_sum"],
                stored["lateral_error_rmse_m"],
                stored["heading_error_rmse_rad"],
                stored["speed_error_rmse_mps"],
            )
            errors[f"{window}_{arm}"] = float(
                np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
            )

    guards = [row["hard_guard"] for row in guard]
    warm = np.asarray([one["warm_cost"] for one in guards])
    proposal = np.asarray([one["proposal_cost"] for one in guards])
    selected = np.asarray([one["selected_cost"] for one in guards])
    selected_index = np.asarray([one["selected_index"] for one in guards])
    hard_min_error = float(np.max(np.abs(selected - np.minimum(warm, proposal))))
    branch_error = int(np.sum(selected_index != (proposal < warm)))
    floor_violation = float(np.max(selected - warm))
    failures = {
        **{name: value for name, value in errors.items() if value > 1e-9},
        **({"hard_min": hard_min_error} if hard_min_error > 1e-6 else {}),
        **({"branch": branch_error} if branch_error else {}),
        **({"floor": floor_violation} if floor_violation > 1e-6 else {}),
    }
    validation = {
        "qualification": "VALIDATED_MECHANISM_PILOT" if not failures else "FAIL",
        "metric_max_abs_error": errors,
        "hard_min_max_abs_error": hard_min_error,
        "branch_mismatch_count": branch_error,
        "warm_floor_max_violation": floor_violation,
        "failures": failures,
        "test_policy": "formal validation and test remain sealed",
    }
    (args.run_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
