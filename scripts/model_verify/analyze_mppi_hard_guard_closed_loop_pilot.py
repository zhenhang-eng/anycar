#!/usr/bin/env python3
"""Analyze the matched-seed warm-MPPI versus hard-guard closed-loop pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_ROOT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "mppi_hard_guard_closed_loop_pilot_20260813_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/hard_guard_closed_loop_pilot_20260813_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", default="baseline_seed3407")
    parser.add_argument("--guard", default="guard_seed3407")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mature-start", type=int, default=250)
    parser.add_argument("--deadline-s", type=float, default=0.05)
    return parser.parse_args()


def load_episode(root: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode = root / name
    trace_path = episode / "closed_loop_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    manifest = json.loads((episode / "manifest.json").read_text())
    if [row["control_step"] for row in rows] != list(range(len(rows))):
        raise AssertionError(f"{name}: trace is not contiguous from step zero")
    return rows, manifest


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def metrics(rows: list[dict[str, Any]], start: int, deadline: float) -> dict[str, Any]:
    rows = [row for row in rows if row["control_step"] >= start]
    state = np.asarray([row["state"] for row in rows], np.float64)
    current = np.asarray([row["current_action"] for row in rows], np.float64)
    action = np.asarray([row["executed_action"] for row in rows], np.float64)
    frenet = np.asarray([row["frenet_pose"] for row in rows], np.float64)
    reference_speed = np.asarray([row["reference"][0][3] for row in rows], np.float64)
    lateral = frenet[:, 1]
    heading = frenet[:, 2]
    speed_error = state[:, 3] - reference_speed
    action_rate = action - current
    action_second_difference = np.diff(action, n=2, axis=0)
    # Realized one-step diagnostic uses the same tracking/rate weights as MPPI,
    # expressed in Frenet coordinates. It is not the horizon proposal cost.
    realized_stage = (
        5.0 * lateral**2
        + 10.0 * heading**2
        + speed_error**2
        + 0.1 * action_rate[:, 0]**2
        + 0.1 * action_rate[:, 1]**2
    )
    duration = np.asarray([row["controller_duration_s"] for row in rows])
    return {
        "step_count": len(rows),
        "start_step": int(rows[0]["control_step"]),
        "end_step": int(rows[-1]["control_step"]),
        "realized_stage_cost_sum": float(realized_stage.sum()),
        "realized_stage_cost_mean": float(realized_stage.mean()),
        "lateral_error_rmse_m": float(np.sqrt(np.mean(lateral**2))),
        "lateral_error_abs_p95_m": float(np.quantile(np.abs(lateral), 0.95)),
        "lateral_error_abs_max_m": float(np.max(np.abs(lateral))),
        "heading_error_rmse_rad": float(np.sqrt(np.mean(heading**2))),
        "heading_error_abs_p95_rad": float(np.quantile(np.abs(heading), 0.95)),
        "heading_error_abs_max_rad": float(np.max(np.abs(heading))),
        "speed_error_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "acceleration_rate_rms": float(np.sqrt(np.mean(action_rate[:, 0]**2))),
        "steering_rate_rms": float(np.sqrt(np.mean(action_rate[:, 1]**2))),
        "acceleration_second_difference_rms": float(
            np.sqrt(np.mean(action_second_difference[:, 0]**2))
        ),
        "steering_second_difference_rms": float(
            np.sqrt(np.mean(action_second_difference[:, 1]**2))
        ),
        "diagnostic_lateral_over_0p5_fraction": float(np.mean(np.abs(lateral) > 0.5)),
        "diagnostic_heading_over_0p5_fraction": float(np.mean(np.abs(heading) > 0.5)),
        "finite_state_action": bool(np.all(np.isfinite(state)) and np.all(np.isfinite(action))),
        "final_frenet_progress_m": float(frenet[-1, 0]),
        "controller_duration_s": distribution(duration),
        "deadline_miss_fraction": float(np.mean(duration > deadline)),
    }


def relative_change(baseline: dict[str, Any], guard: dict[str, Any]) -> dict[str, float]:
    keys = (
        "realized_stage_cost_sum",
        "lateral_error_rmse_m",
        "heading_error_rmse_rad",
        "speed_error_rmse_mps",
        "acceleration_rate_rms",
        "steering_rate_rms",
        "acceleration_second_difference_rms",
        "steering_second_difference_rms",
    )
    return {
        key: float((guard[key] - baseline[key]) / max(abs(baseline[key]), 1e-12))
        for key in keys
    }


def guard_selection(rows: list[dict[str, Any]], start: int) -> dict[str, Any]:
    guards = [row["hard_guard"] for row in rows if row["control_step"] >= start]
    selected = np.asarray([one["selected"] == "proposal" for one in guards])
    warm = np.asarray([one["warm_cost"] for one in guards], np.float64)
    proposal = np.asarray([one["proposal_cost"] for one in guards], np.float64)
    chosen = np.asarray([one["selected_cost"] for one in guards], np.float64)
    transitions = int(np.sum(selected[1:] != selected[:-1]))
    run_lengths = []
    begin = 0
    for index in range(1, len(selected) + 1):
        if index == len(selected) or selected[index] != selected[begin]:
            run_lengths.append((bool(selected[begin]), index - begin))
            begin = index
    runtime = np.asarray(
        [one["actor_runtime_duration_s"] for one in guards], np.float64
    )
    return {
        "step_count": len(guards),
        "proposal_selected_fraction": float(selected.mean()),
        "warm_selected_fraction": float(1.0 - selected.mean()),
        "selection_transition_count": transitions,
        "selection_transition_fraction": transitions / max(len(selected) - 1, 1),
        "longest_proposal_run": max((n for flag, n in run_lengths if flag), default=0),
        "longest_warm_run": max((n for flag, n in run_lengths if not flag), default=0),
        "warm_model_cost": distribution(warm),
        "proposal_model_cost": distribution(proposal),
        "selected_model_cost": distribution(chosen),
        "model_cost_gain_vs_warm": distribution(warm - chosen),
        "warm_floor_max_violation": float(np.max(chosen - warm)),
        "actor_runtime_duration_s": distribution(runtime),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline_rows, baseline_manifest = load_episode(args.root, args.baseline)
    guard_rows, guard_manifest = load_episode(args.root, args.guard)
    if len(baseline_rows) != len(guard_rows):
        raise AssertionError("A/B trace lengths differ")
    for key in ("mppi_params", "cost_weights_at_collection"):
        if baseline_manifest[key] != guard_manifest[key]:
            raise AssertionError(f"A/B {key} mismatch")
    if baseline_manifest["collection"]["reference_speed_override_mps"] != guard_manifest["collection"]["reference_speed_override_mps"]:
        raise AssertionError("A/B reference speed mismatch")

    windows = {}
    for name, start in (("all_steps", 0), ("history_complete", args.mature_start)):
        baseline = metrics(baseline_rows, start, args.deadline_s)
        guard = metrics(guard_rows, start, args.deadline_s)
        windows[name] = {
            "baseline": baseline,
            "guard": guard,
            "guard_relative_change": relative_change(baseline, guard),
            "guard_selection": guard_selection(guard_rows, start),
        }

    summary = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "MECHANISM_PILOT_ONLY",
        "root": str(args.root.resolve()),
        "baseline_episode": args.baseline,
        "guard_episode": args.guard,
        "baseline_manifest_sha256": sha256_file(args.root / args.baseline / "manifest.json"),
        "guard_manifest_sha256": sha256_file(args.root / args.guard / "manifest.json"),
        "matched_contract": {
            "mppi_seed": baseline_manifest["mppi_params"]["seed"],
            "num_samples": baseline_manifest["mppi_params"]["num_samples"],
            "sampling_mode": baseline_manifest["mppi_params"]["sampling_mode"],
            "reference_speed_mps": baseline_manifest["collection"]["reference_speed_override_mps"],
            "control_dt_s": baseline_manifest["mppi_params"]["dt"],
            "trace_steps": len(baseline_rows),
            "guard_rollouts_per_step": 387,
        },
        "windows": windows,
        "test_policy": "formal validation and test remain sealed",
        "caveats": [
            "One matched seed and one 2.8-m/s scene are a mechanism pilot, not a statistical closed-loop qualification.",
            "Realized stage cost is a Frenet one-step diagnostic, not MPPI horizon proposal cost.",
            "The step-mode simulator preserves simulated dt despite controller wall-clock overruns.",
            "0.5-m/rad exceedance rates are diagnostics, not a certified failure definition.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
