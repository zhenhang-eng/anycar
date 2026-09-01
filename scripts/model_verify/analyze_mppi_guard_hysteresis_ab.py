#!/usr/bin/env python3
"""Analyze the guard hysteresis closed-loop A/B (27 episodes).

Paired by seed within each scenario: baseline vs guard (plain argmin) vs
guard+hysteresis (margin 1.0, dwell 5). Metrics reuse the pilot analyzer
conventions (realized Frenet tracking/rate stage cost, RMSEs, control rate
and second-difference RMS, guard selection statistics). Hysteresis can hold
an incumbent whose cost exceeds the warm branch, so warm-floor violations
are tracked explicitly rather than assumed zero.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analyze_mppi_hard_guard_closed_loop_pilot import metrics

DEFAULT_ROOT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "guard_hysteresis_ab_20260818_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/guard_hysteresis_ab_20260818_v1")
SCENARIOS = ("nominal", "high", "recovery")
ARMS = ("baseline", "guard", "guard_hyst")
SEEDS = (3407, 3411, 3413)
MATURE_START = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_episode(root: Path, name: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / name / "closed_loop_trace.jsonl")
        .read_text()
        .splitlines()
    ]
    if [row["control_step"] for row in rows] != list(range(len(rows))):
        raise AssertionError(f"{name}: trace not contiguous")
    return rows


def guard_stats(rows: list[dict], start: int) -> dict:
    guards = [row["hard_guard"] for row in rows if row["control_step"] >= start]
    if not guards:
        return {}
    selected = np.asarray([g["selected"] == "proposal" for g in guards])
    warm = np.asarray([g["warm_cost"] for g in guards], np.float64)
    chosen = np.asarray([g["selected_cost"] for g in guards], np.float64)
    switches = int(np.sum(selected[1:] != selected[:-1]))
    return {
        "actor_rate": float(np.mean(selected)),
        "switch_rate": float(switches / max(len(selected) - 1, 1)),
        "warm_floor_violation_fraction": float(np.mean(chosen > warm + 1e-9)),
        "floor_violation_cost_mean": float(
            np.mean(np.maximum(chosen - warm, 0.0))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {}
    for scenario in SCENARIOS:
        scenario_rows = {}
        for arm in ARMS:
            per_seed = {}
            for seed in SEEDS:
                name = f"{arm}_{scenario}_s{seed}"
                rows = load_episode(args.root, name)
                per_seed[str(seed)] = {
                    "full": metrics(rows, 0, 0.05),
                    "mature": metrics(rows, MATURE_START, 0.05),
                }
                if arm != "baseline":
                    per_seed[str(seed)]["guard_full"] = guard_stats(rows, 0)
                    per_seed[str(seed)]["guard_mature"] = guard_stats(
                        rows, MATURE_START
                    )
            scenario_rows[arm] = per_seed

        def paired(comparison: str, base: str, challenger: str, key: str,
                   field: str) -> dict:
            values = []
            for seed in SEEDS:
                b = scenario_rows[base][str(seed)][key][field]
                c = scenario_rows[challenger][str(seed)][key][field]
                values.append(
                    (c - b) / max(abs(b), 1e-12)
                )
            return {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        comparisons = {}
        for label, base, challenger in (
            ("guard_vs_baseline", "baseline", "guard"),
            ("hyst_vs_baseline", "baseline", "guard_hyst"),
            ("hyst_vs_guard", "guard", "guard_hyst"),
        ):
            entry = {}
            for key in ("full", "mature"):
                entry[key] = {
                    field: paired(label, base, challenger, key, field)
                    for field in (
                        "realized_stage_cost_sum",
                        "lateral_error_rmse_m",
                        "heading_error_rmse_rad",
                        "speed_error_rmse_mps",
                        "acceleration_rate_rms",
                        "steering_rate_rms",
                        "acceleration_second_difference_rms",
                        "steering_second_difference_rms",
                    )
                }
            comparisons[label] = entry

        guard_summary = {}
        for arm in ("guard", "guard_hyst"):
            for key, label in (("guard_full", "full"), ("guard_mature", "mature")):
                entry = {}
                for field in (
                    "actor_rate", "switch_rate",
                    "warm_floor_violation_fraction",
                ):
                    entry[field] = float(np.median([
                        scenario_rows[arm][str(seed)][key][field]
                        for seed in SEEDS
                    ]))
                guard_summary[f"{arm}_{label}"] = entry

        report[scenario] = {
            "absolute": scenario_rows,
            "comparisons": comparisons,
            "guard_summary": guard_summary,
        }

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GUARD_HYSTERESIS_AB_ANALYZED_ACTOR_FROZEN",
        "sources": {"root": str(args.root)},
        "protocol": {
            "episodes": len(SCENARIOS) * len(ARMS) * len(SEEDS),
            "hysteresis": "margin 1.0, dwell 5 (pre-registered first round)",
            "mature_start": MATURE_START,
        },
        "report": report,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    for scenario in SCENARIOS:
        print(f"== {scenario} ==")
        for label in ("guard_vs_baseline", "hyst_vs_baseline", "hyst_vs_guard"):
            full = report[scenario]["comparisons"][label]["full"][
                "realized_stage_cost_sum"
            ]
            mature = report[scenario]["comparisons"][label]["mature"][
                "realized_stage_cost_sum"
            ]
            rate = report[scenario]["comparisons"][label]["full"][
                "acceleration_second_difference_rms"
            ]
            print(
                f"  {label}: stage full {full['median']:+.3f} "
                f"[{full['min']:+.3f},{full['max']:+.3f}] | mature "
                f"{mature['median']:+.3f} | acc2diff {rate['median']:+.3f}"
            )
        for arm in ("guard", "guard_hyst"):
            g = report[scenario]["guard_summary"][f"{arm}_mature"]
            print(
                f"  {arm} mature: actor {g['actor_rate']:.3f} "
                f"switch {g['switch_rate']:.3f} "
                f"floorviol {g['warm_floor_violation_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
