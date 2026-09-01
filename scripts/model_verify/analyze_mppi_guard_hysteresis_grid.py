#!/usr/bin/env python3
"""Guard hysteresis margin/dwell grid analysis (Pareto view).

For each (margin, dwell) cell, per scenario (seed medians):
- safety cost = mature warm-floor violation fraction x mean excess cost;
- benefit = mature realized stage cost improvement vs the warm-only baseline
  episodes from the parent A/B (full-window improvement also reported);
- every cell is checked against the warm-only reference so a dominated cell
  is flagged instead of silently kept.
- per-step duration decomposition: total controller duration, the actor
  129-rollout context duration, and the remainder dominated by the warm 256
  rollouts plus the two guard evaluations (starting point data for any
  future realtime work).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analyze_mppi_guard_hysteresis_ab import (
    ARMS,
    SCENARIOS,
    SEEDS,
    guard_stats,
    load_episode,
)
from analyze_mppi_hard_guard_closed_loop_pilot import metrics

DEFAULT_ROOT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "guard_hysteresis_ab_20260818_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/guard_hysteresis_grid_20260818_v1")
GRID = (
    (0.5, 3), (0.5, 5), (0.5, 8),
    (1.0, 5),
    (2.0, 3), (2.0, 5), (2.0, 8),
    ("1.0whr", 5),
)
MATURE_START = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def episode_id(margin, dwell: int, scenario: str, seed: int) -> str:
    if margin == "1.0whr":
        return f"ghist_m1.0_d5whr_{scenario}_s{seed}"
    if (margin, dwell) == (1.0, 5):
        return f"guard_hyst_{scenario}_s{seed}"
    return f"ghist_m{margin}_d{dwell}_{scenario}_s{seed}"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline_stage = {}
    for scenario in SCENARIOS:
        values = []
        for seed in SEEDS:
            rows = load_episode(args.root, f"baseline_{scenario}_s{seed}")
            mature = metrics(rows, MATURE_START, 0.05)
            full = metrics(rows, 0, 0.05)
            values.append((full["realized_stage_cost_sum"],
                           mature["realized_stage_cost_sum"]))
        baseline_stage[scenario] = {
            "full_median": float(np.median([v[0] for v in values])),
            "mature_median": float(np.median([v[1] for v in values])),
        }

    report = {}
    for margin, dwell in GRID:
        cell = {}
        for scenario in SCENARIOS:
            mature_stage, full_stage = [], []
            violations, excess, actor, switches = [], [], [], []
            durations, actor_durations = [], []
            for seed in SEEDS:
                name = episode_id(margin, dwell, scenario, seed)
                rows = load_episode(args.root, name)
                full = metrics(rows, 0, 0.05)
                mature = metrics(rows, MATURE_START, 0.05)
                full_stage.append(full["realized_stage_cost_sum"])
                mature_stage.append(mature["realized_stage_cost_sum"])
                guard_mature = guard_stats(rows, MATURE_START)
                violations.append(guard_mature["warm_floor_violation_fraction"])
                excess.append(guard_mature["floor_violation_cost_mean"])
                actor.append(guard_mature["actor_rate"])
                switches.append(guard_mature["switch_rate"])
                durations.extend([
                    row["controller_duration_s"] for row in rows
                ])
                actor_durations.extend([
                    row["hard_guard"]["actor_runtime_duration_s"]
                    for row in rows if "hard_guard" in row
                ])
            mature_median = float(np.median(mature_stage))
            full_median = float(np.median(full_stage))
            violation_median = float(np.median(violations))
            excess_median = float(np.median(excess))
            cell[scenario] = {
                "mature_stage_improvement": float(
                    (baseline_stage[scenario]["mature_median"] - mature_median)
                    / baseline_stage[scenario]["mature_median"]
                ),
                "full_stage_improvement": float(
                    (baseline_stage[scenario]["full_median"] - full_median)
                    / baseline_stage[scenario]["full_median"]
                ),
                "safety_cost": violation_median * excess_median,
                "violation_fraction": violation_median,
                "mean_excess_cost": excess_median,
                "actor_rate": float(np.median(actor)),
                "switch_rate": float(np.median(switches)),
                "beats_baseline_mature": bool(
                    mature_median
                    < baseline_stage[scenario]["mature_median"]
                ),
                "duration_decomposition_s": {
                    "total_p50": float(np.quantile(durations, 0.50)),
                    "total_p95": float(np.quantile(durations, 0.95)),
                    "actor_runtime_p50": float(
                        np.quantile(actor_durations, 0.50)
                    ),
                    "remainder_p50": float(
                        np.quantile(durations, 0.50)
                        - np.quantile(actor_durations, 0.50)
                    ),
                },
            }
        report[f"m{margin}_d{dwell}"] = cell

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GUARD_HYSTERESIS_GRID_ANALYZED_ACTOR_FROZEN",
        "sources": {"root": str(args.root)},
        "protocol": {
            "grid": [list(cell) for cell in GRID],
            "seeds": list(SEEDS),
            "mature_start": MATURE_START,
            "safety_cost": "violation_fraction x mean_excess_cost",
            "note": "m1.0_d5 cells reuse the guard_hyst arm of the parent A/B",
        },
        "baseline_stage": baseline_stage,
        "report": report,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    for scenario in SCENARIOS:
        print(f"== {scenario} (mature improvement | safety | viol% | beats_base) ==")
        for key, cell in report.items():
            c = cell[scenario]
            print(
                f"  {key:10s} {c['mature_stage_improvement']:+.3f} | "
                f"{c['safety_cost']:.3f} | {c['violation_fraction']*100:5.1f}% | "
                f"{c['beats_baseline_mature']}"
            )


if __name__ == "__main__":
    main()
