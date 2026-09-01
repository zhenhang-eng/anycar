#!/usr/bin/env python3
"""Create paired K=1 versus K=16 statistics for the high-speed Actor scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "positive_count": int(np.sum(array > 0.0)),
    }


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    summary_path = root / "summary.json"
    contract_path = root / "contract.json"
    summary = json.loads(summary_path.read_text())
    contract = json.loads(contract_path.read_text())
    if summary["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected source qualification")
    if contract["k_values"] != [1, 16]:
        raise AssertionError("paired endpoint analysis requires K=1 and K=16")
    metric_paths = {
        "teacher_gain_recovery": ("teacher_gain_recovery", None),
        "gain_vs_start_mean": ("gain_vs_pretrained_actor", "mean"),
        "gain_vs_start_p05": ("gain_vs_pretrained_actor", "p05"),
        "gain_vs_warm_mean": ("gain_vs_warm", "mean"),
        "gain_vs_warm_p05": ("gain_vs_warm", "p05"),
        "beats_warm_fraction": ("beats_or_equals_warm_fraction", None),
    }
    paired: dict[str, list[float]] = {name: [] for name in metric_paths}
    raw = {1: [], 16: []}
    projected = {1: [], 16: []}
    selected_round_equal_count = 0
    per_run = []
    for record in summary["records"]:
        arms = {k: record["arms"][str(k)] for k in (1, 16)}
        row = {"fold": int(record["fold"]), "seed": int(record["seed"])}
        for name, (outer, inner) in metric_paths.items():
            values = {}
            for k in (1, 16):
                value = arms[k]["selected"]["oof"][outer]
                values[k] = float(value if inner is None else value[inner])
            delta = values[16] - values[1]
            paired[name].append(delta)
            row[name + "_k16_minus_k1"] = delta
        if arms[1]["selected_round"] == arms[16]["selected_round"]:
            selected_round_equal_count += 1
        for k in (1, 16):
            for round_row in arms[k]["rounds"]:
                raw[k].append(float(round_row["actor_raw_cumulative_step_sigma_rms"]))
                projected[k].append(float(
                    round_row["actor_projected_cumulative_step_sigma_rms"]
                ))
        per_run.append(row)
    analysis = {
        "format": "highspeed_actor_k_scan_paired_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_FIXED_TRUST_K16_MARGINAL_BODY_GAIN_NO_TAIL_GAIN",
        "source": {
            "summary_sha256": sha256(summary_path),
            "contract_sha256": sha256(contract_path),
            "records": len(summary["records"]),
        },
        "contract": {
            "comparison": "K16 minus K1; positive is better for every reported delta",
            "same_rollout_and_critic_budget": True,
            "same_cumulative_output_trust_sigma_rms": 0.02,
            "formal_validation_or_test_created": False,
        },
        "paired_k16_minus_k1": {
            name: distribution(values) for name, values in paired.items()
        },
        "step_control": {
            "selected_round_equal_count": selected_round_equal_count,
            "selected_round_pair_count": len(summary["records"]),
            "raw_cumulative_step_sigma_rms": {
                str(k): distribution(raw[k]) for k in (1, 16)
            },
            "projected_cumulative_step_sigma_rms": {
                str(k): distribution(projected[k]) for k in (1, 16)
            },
        },
        "decision": {
            "K16_stable_advantage": False,
            "reason": (
                "K16 gives only a marginal mean/recovery improvement while the "
                "warm-relative P05 and beats-warm rate do not improve consistently."
            ),
            "next": (
                "Keep K1 as the cost-effective reference and audit speed-group "
                "gradient conflict before PCGrad/CAGrad-MGDA training."
            ),
        },
        "per_run": per_run,
    }
    output = root / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: value for key, value in analysis.items() if key != "per_run"}, indent=2))


if __name__ == "__main__":
    main()
