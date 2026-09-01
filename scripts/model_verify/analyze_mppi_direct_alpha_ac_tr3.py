#!/usr/bin/env python3
"""Summarize frozen TR3 move/stay and endpoint-tail failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_mppi_direct_alpha_ac_tr3 import DEFAULT_OUTPUT, distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.run.glob("episode_*/*.npz")):
        with np.load(path, allow_pickle=False) as value:
            for context in range(len(value["old_center"])):
                safe = int(value["safe_index"][context])
                old = float(value["line_cost"][context, 0])
                tr2b = float(value["tr2b_cost"][context])
                alpha_ac = float(value["alpha_ac_cost"][context])
                rows.append({
                    "key": f"{path.parent.name}/{path.stem}/context_{context}",
                    "episode": path.parent.name,
                    "reference_speed_mps": float(value["reference_speed_mps"]),
                    "scenario": str(value["scenario"]),
                    "old_cost": old,
                    "tr2b_cost": tr2b,
                    "alpha_ac_cost": alpha_ac,
                    "tr2b_gain_vs_old": old - tr2b,
                    "alpha_ac_gain_vs_old": old - alpha_ac,
                    "alpha_ac_gain_vs_tr2b": tr2b - alpha_ac,
                    "safe_index": safe,
                    "safe_alpha": float(value["alpha_grid"][safe]),
                    "tr2b_probability": float(value["tr2b_probability"][context]),
                    "tr2b_conditional_alpha": float(value["tr2b_conditional_alpha"][context]),
                    "tr2b_alpha": float(value["tr2b_alpha"][context]),
                    "alpha_ac_probability": float(value["alpha_ac_probability"][context]),
                    "alpha_ac_conditional_alpha": float(value["alpha_ac_conditional_alpha"][context]),
                    "alpha_ac_alpha": float(value["alpha_ac_alpha"][context]),
                })

    def policy(name: str) -> dict:
        gain = np.asarray([row[f"{name}_gain_vs_old"] for row in rows])
        alpha = np.asarray([row[f"{name}_alpha"] for row in rows])
        conditional = np.asarray([row[f"{name}_conditional_alpha"] for row in rows])
        safe_zero = np.asarray([row["safe_index"] == 0 for row in rows])
        move = alpha > 0.0
        return {
            "gain_vs_old": distribution(gain),
            "regression_count": int(np.sum(gain < 0.0)),
            "regression_below_minus_5_count": int(np.sum(gain < -5.0)),
            "move_count": int(np.sum(move)),
            "move_fraction": float(np.mean(move)),
            "move_when_safe_zero_count": int(np.sum(move & safe_zero)),
            "move_when_safe_zero_regression_count": int(np.sum(move & safe_zero & (gain < 0.0))),
            "conditional_alpha_on_move": distribution(conditional[move]),
        }

    patterns = {}
    for name, predicate in {
        "both_stay": lambda row: row["tr2b_alpha"] == 0 and row["alpha_ac_alpha"] == 0,
        "alpha_ac_only_moves": lambda row: row["tr2b_alpha"] == 0 and row["alpha_ac_alpha"] > 0,
        "both_move": lambda row: row["tr2b_alpha"] > 0 and row["alpha_ac_alpha"] > 0,
        "tr2b_only_moves": lambda row: row["tr2b_alpha"] > 0 and row["alpha_ac_alpha"] == 0,
    }.items():
        selected = [row for row in rows if predicate(row)]
        gain_old = np.asarray([row["alpha_ac_gain_vs_old"] for row in selected])
        gain_tr2b = np.asarray([row["alpha_ac_gain_vs_tr2b"] for row in selected])
        patterns[name] = {
            "context_count": len(selected),
            "alpha_ac_gain_vs_old": distribution(gain_old) if len(selected) else None,
            "alpha_ac_gain_vs_tr2b": distribution(gain_tr2b) if len(selected) else None,
        }

    report = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "context_count": len(rows),
        "tr2b": policy("tr2b"),
        "alpha_ac": policy("alpha_ac"),
        "move_pattern": patterns,
        "worst_alpha_ac_contexts": sorted(
            rows, key=lambda row: row["alpha_ac_gain_vs_old"]
        )[:20],
        "diagnosis": [
            "both learned alpha policies fail the frozen formal worst-tail gate",
            "every Alpha-AC move whose formal safe alpha is zero regresses",
            "conditional alpha on moved contexts is nearly saturated at one",
            "the learned policy behaves mainly as stay-versus-endpoint selection, not calibrated continuous step selection",
            "validation must not be reused to retune threshold or checkpoint",
        ],
        "qualification": "TR3_DIAGNOSIS_COMPLETE_RETURN_TO_TRAIN_ONLY_RISK_AND_STEP_DATA",
        "test_policy": "test episodes 105--119 not opened",
    }
    (args.run / "tail_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
