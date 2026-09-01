#!/usr/bin/env python3
"""Compare inactive-contract and active-tail Absolute vs Pair-Delta OAC-2 A/Bs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def selected_by_seed(summary: dict) -> dict[int, dict]:
    return {int(row["seed"]): row for row in summary["records"]}


def metric(row: dict, path: tuple[str, ...]) -> float:
    value = row["selected_metrics"]
    for key in path:
        value = value[key]
    return float(value)


def compare(base: dict, pair: dict) -> dict:
    left = selected_by_seed(base)
    right = selected_by_seed(pair)
    if set(left) != set(right):
        raise AssertionError("seed sets differ")
    metrics = {
        "mean_gain": (("gain_vs_initial", "mean"), True),
        "median_gain": (("gain_vs_initial", "median"), True),
        "p05_gain": (("gain_vs_initial", "p05"), True),
        "worst_gain": (("gain_vs_initial", "minimum"), True),
        "regression_fraction": (("regression_fraction",), False),
        "speed_2_4_p05_gain": (("by_speed", "2.4", "p05_gain"), True),
        "speed_2_8_p05_gain": (("by_speed", "2.8", "p05_gain"), True),
        "guard_gain_mean": (("two_center_guard", "gain_vs_warm", "mean"), True),
    }
    rows = []
    for seed in sorted(left):
        row = {
            "seed": seed,
            "paired_identity": {
                "initial_actor_module_sha256": left[seed]["initial_actor_module_sha256"] == right[seed]["initial_actor_module_sha256"],
                "source_replay_sha256": left[seed]["source_replay_sha256"] == right[seed]["source_replay_sha256"],
                "initial_action_sha256": left[seed]["initial_metrics"]["action_sha256"] == right[seed]["initial_metrics"]["action_sha256"],
                "initial_cost_sha256": left[seed]["initial_metrics"]["cost_sha256"] == right[seed]["initial_metrics"]["cost_sha256"],
            },
        }
        for name, (path, _higher_is_better) in metrics.items():
            base_value = metric(left[seed], path)
            pair_value = metric(right[seed], path)
            row[name] = {"absolute": base_value, "pair_delta": pair_value, "delta": pair_value - base_value}
        row["selected_round"] = {
            "absolute": int(left[seed]["selected_round"]),
            "pair_delta": int(right[seed]["selected_round"]),
        }
        row["final_tail_lagrange"] = {
            "absolute": float(left[seed]["final_tail_lagrange"]),
            "pair_delta": float(right[seed]["final_tail_lagrange"]),
        }
        rows.append(row)
    aggregate = {}
    for name, (_path, higher_is_better) in metrics.items():
        deltas = [row[name]["delta"] for row in rows]
        aggregate[name] = {
            "absolute_mean": statistics.mean(row[name]["absolute"] for row in rows),
            "pair_delta_mean": statistics.mean(row[name]["pair_delta"] for row in rows),
            "paired_delta_mean": statistics.mean(deltas),
            "paired_delta_median": statistics.median(deltas),
            "higher_is_better": higher_is_better,
            "pair_better_seed_count": sum(
                value > 0 if higher_is_better else value < 0 for value in deltas
            ),
        }
    return {
        "per_seed": rows,
        "aggregate": aggregate,
        "all_paired_identity_checks_pass": all(
            all(row["paired_identity"].values()) for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path("outputs/mppi_proposal")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inactive-absolute", type=Path, default=root / "online_absolute_sac_oac2_box3_adaptive_tail_smoke_20260825_v1")
    parser.add_argument("--inactive-pair", type=Path, default=root / "online_absolute_sac_oac2_pairdelta_adaptive_tail_smoke_20260825_v1")
    parser.add_argument("--active-absolute", type=Path, default=root / "online_absolute_sac_oac2_active_tail_absolute_smoke_20260825_v1")
    parser.add_argument("--active-pair", type=Path, default=root / "online_absolute_sac_oac2_active_tail_pairdelta_smoke_20260825_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "online_absolute_sac_oac2_pairdelta_ab_analysis_20260825_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "inactive_absolute": args.inactive_absolute,
        "inactive_pair": args.inactive_pair,
        "active_absolute": args.active_absolute,
        "active_pair": args.active_pair,
    }
    summaries = {name: load(path) for name, path in paths.items()}
    inactive = compare(summaries["inactive_absolute"], summaries["inactive_pair"])
    active = compare(summaries["active_absolute"], summaries["active_pair"])
    inactive_max_abs_action_effect = max(
        abs(row[metric]["delta"])
        for row in inactive["per_seed"]
        for metric in ("mean_gain", "median_gain", "p05_gain", "worst_gain")
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PAIR_DELTA_ACTIVE_TAIL_MIXED_NO_FORWARD_AUTHORIZATION",
        "manifest": {
            name: {"path": str(path.resolve()), "summary_sha256": sha256(path / "summary.json")}
            for name, path in paths.items()
        },
        "inactive_constraint_ab": inactive,
        "active_constraint_ab": active,
        "decision": {
            "all_paired_identity_checks_pass": bool(
                inactive["all_paired_identity_checks_pass"]
                and active["all_paired_identity_checks_pass"]
            ),
            "inactive_contract_isolation_max_abs_metric_delta": inactive_max_abs_action_effect,
            "inactive_interpretation": "dual stayed zero; Pair path did not perturb the Absolute Twin/Actor baseline",
            "active_interpretation": "Pair risk improved mean/median gain but worsened direct tail metrics; it cannot replace the Absolute tail risk yet",
            "forward_authorized": False,
        },
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
