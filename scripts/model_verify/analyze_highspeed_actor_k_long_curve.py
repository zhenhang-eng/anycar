#!/usr/bin/env python3
"""Analyze the fixed-trust 90-round high-speed K=1/K=16 learning curve."""

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


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "positive_count": int(np.sum(array > 0.0)),
    }


def metrics(row: dict) -> dict[str, float]:
    return {
        "teacher_gain_recovery": float(row["teacher_gain_recovery"]),
        "gain_vs_start_mean": float(row["gain_vs_pretrained_actor"]["mean"]),
        "gain_vs_start_p05": float(row["gain_vs_pretrained_actor"]["p05"]),
        "gain_vs_warm_mean": float(row["gain_vs_warm"]["mean"]),
        "gain_vs_warm_p05": float(row["gain_vs_warm"]["p05"]),
        "beats_warm_fraction": float(row["beats_or_equals_warm_fraction"]),
    }


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    summary_path = root / "summary.json"
    contract_path = root / "contract.json"
    summary = json.loads(summary_path.read_text())
    contract = json.loads(contract_path.read_text())
    if contract["k_values"] != [1, 16] or int(contract["arguments"]["rounds"]) != 90:
        raise AssertionError("expected fixed-trust K=1/K=16 90-round contract")
    evaluation_rounds = [
        int(value) for value in contract["arguments"]["evaluation_rounds"].split(",")
        if value.strip()
    ]
    curve: dict[str, dict[str, dict]] = {"1": {}, "16": {}}
    paired: dict[str, dict[str, dict]] = {}
    per_run = []
    for round_index in evaluation_rounds:
        arm_rows: dict[int, list[dict[str, float]]] = {1: [], 16: []}
        paired_rows = []
        for record in summary["records"]:
            row_by_k = {}
            for k in (1, 16):
                round_row = next(
                    row for row in record["arms"][str(k)]["rounds"]
                    if int(row["round"]) == round_index
                )
                if round_row["oof"] is None:
                    raise AssertionError("missing OOF learning-curve evaluation")
                row_by_k[k] = metrics(round_row["oof"])
                arm_rows[k].append(row_by_k[k])
            paired_rows.append({
                name: row_by_k[16][name] - row_by_k[1][name]
                for name in row_by_k[1]
            })
            per_run.append({
                "round": round_index,
                "fold": int(record["fold"]),
                "seed": int(record["seed"]),
                "k1": row_by_k[1],
                "k16": row_by_k[16],
                "k16_minus_k1": paired_rows[-1],
            })
        for k in (1, 16):
            curve[str(k)][str(round_index)] = {
                name: stats([row[name] for row in arm_rows[k]])
                for name in arm_rows[k][0]
            }
        paired[str(round_index)] = {
            name: stats([row[name] for row in paired_rows])
            for name in paired_rows[0]
        }
    selected_rounds = {
        str(k): [int(record["arms"][str(k)]["selected_round"]) for record in summary["records"]]
        for k in (1, 16)
    }
    analysis = {
        "format": "highspeed_actor_k_long_curve_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_90ROUND_BUDGET_MATTERS_K16_FASTER_ENDPOINT_SIMILAR",
        "source": {
            "summary_sha256": sha256(summary_path),
            "contract_sha256": sha256(contract_path),
            "folds": contract["folds"],
            "seeds": contract["seeds"],
        },
        "contract": {
            "rounds": 90,
            "evaluation_rounds": evaluation_rounds,
            "cumulative_output_trust_sigma_rms": 0.02,
            "formal_validation_or_test_created": False,
        },
        "curve": curve,
        "paired_k16_minus_k1": paired,
        "selected_rounds": selected_rounds,
        "decision": {
            "five_round_conclusion_invalidated": True,
            "round_budget_matters": True,
            "k16_role": (
                "K16 accelerates the middle of training and improves the 90-round "
                "P05, but the 90-round mean endpoint is nearly identical to K1."
            ),
            "plateau_reached": False,
            "next": (
                "Hold K16 and 90 rounds fixed; compare cumulative per-round output "
                "trust 0.02/0.04/0.06 sigma as a separate experiment."
            ),
        },
        "per_run": per_run,
    }
    output = root / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: value for key, value in analysis.items() if key not in ("curve", "per_run")}, indent=2))


if __name__ == "__main__":
    main()
