#!/usr/bin/env python3
"""Zero-rollout norm-gated stay analysis for the stationarity step audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analyze_mppi_stationarity_step_ab import ROOTS, DEFAULT_OUTPUT, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", default="0.5,1.0,2.0")
    parser.add_argument("--eta", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = json.loads((args.step_root / "analysis.json").read_text())
    with np.load(args.step_root / "per_state.npz", allow_pickle=False) as stored:
        arrays = {key: np.asarray(stored[key]) for key in stored.files}
    thresholds = [float(value) for value in args.thresholds.split(",")]
    rows = []
    for row in analysis["rows"]:
        if row["eta"] != args.eta:
            continue
        arm, seed, anchor = row["arm"], int(row["seed"]), row["anchor"]
        key = f"{arm}_seed{seed}_{anchor}_eta_{args.eta:g}".replace(".", "p")
        gain = arrays[f"{key}_gain"]
        gradient = arrays[f"{arm}_seed{seed}_gradient"]
        anchor_index = 0 if anchor == "warm" else 2
        norm = np.linalg.norm(gradient[:, anchor_index].reshape(len(gain), -1), axis=1)
        for threshold in thresholds:
            move = norm >= threshold
            gated_gain = np.where(move, gain, 0.0)
            rows.append({
                "arm": arm, "seed": seed, "anchor": anchor,
                "eta": args.eta, "gradient_norm_threshold": threshold,
                "stay_fraction": float(np.mean(~move)),
                "gated_gain": summary(gated_gain),
                "moving_subset_gain": summary(gain[move]) if np.any(move) else None,
                "gradient_norm": summary(norm),
            })
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "STATIONARITY_SUPPRESSES_FLAT_REGION_UPDATES_"
            "BUT_GRADIENT_ACTOR_STILL_REQUIRES_TRUE_COST_GUARD"
        ),
        "contract": {
            "source": str(args.step_root / "per_state.npz"),
            "eta": args.eta, "thresholds": thresholds,
            "rule": "stay iff predicted absolute-action log-value gradient norm < threshold",
            "formal_validation_test": "not read; sealed",
        },
        "decision": {
            "selected_mechanism_arm": "S1_lambda_0.1",
            "strong_arm": "not expanded: it suppresses useful warm/raw response",
            "norm_gate_role": (
                "separates most flat best candidates from warm, but does not "
                "remove rare high-norm harmful best-candidate outliers"
            ),
            "actor_gradient_policy": (
                "do not authorize standalone SAC-style updates; bounded training "
                "steps require DBM/Query accept-reject and deployment keeps the "
                "two-center warm guard"
            ),
        },
        "rows": rows,
    }
    output = args.step_root / "norm_gate_analysis.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        if row["gradient_norm_threshold"] == 1.0:
            print(
                row["arm"], row["seed"], row["anchor"],
                f"stay={row['stay_fraction']:.3f}",
                f"mean={row['gated_gain']['mean']:.4f}",
                f"p05={row['gated_gain']['p05']:.4f}",
                f"worst={row['gated_gain']['worst']:.4f}",
            )


if __name__ == "__main__":
    main()
