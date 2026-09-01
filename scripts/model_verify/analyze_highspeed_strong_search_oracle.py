#!/usr/bin/env python3
"""Summarize center baselines and start-conditioned recovery of strong search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v1")
SIGMA = np.asarray((0.25, 0.35), np.float64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = root / "summary.json"
    validator_path = root / "validator_report.json"
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    if validator["qualification"] != "HIGHSPEED_STRONG_SEARCH_ORACLE_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("oracle did not pass independent replay")
    artifact_path = Path(summary["artifact"])
    if sha256(artifact_path) != summary["artifact_sha256"]:
        raise AssertionError("oracle artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    names = data["start_names"].astype(str)
    start_cost = data["start_costs"].astype(np.float64)
    terminal_cost = data["terminal_costs"].astype(np.float64)
    warm = start_cost[:, 0]
    oracle = data["oracle_costs"].astype(np.float64)
    oracle_gain = warm - oracle

    def metrics(name: str, cost: np.ndarray) -> dict:
        gain = warm - cost
        return {
            "name": name, "cost": distribution(cost), "gain": distribution(gain),
            "warm_relative_reduction": float(gain.sum() / warm.sum()),
            "strong_oracle_headroom_recovery": float(gain.sum() / oracle_gain.sum()),
            "beats_or_equals_warm_fraction": float(np.mean(gain >= 0.0)),
        }

    rows = [metrics(name, start_cost[:, index]) for index, name in enumerate(names)]
    rows.append(metrics("best_raw_start", start_cost.min(axis=1)))
    for index in range(2, 5):
        rows.append(metrics(f"warm_plus_{names[index]}", np.minimum(warm, start_cost[:, index])))
    for index, name in enumerate(names):
        rows.append(metrics(f"search_from_{name}", terminal_cost[:, index]))
    rows.append(metrics("strong_oracle", oracle))
    winning_start = np.argmin(terminal_cost, axis=1)
    paired_residual = {}
    for index in range(2, 5):
        delta = (data["terminal_centers"][:, index] - data["start_centers"][:, index]) / SIGMA
        rms = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
        paired_residual[names[index]] = distribution(rms)
    actor_guard_recovery = np.asarray([
        next(row["strong_oracle_headroom_recovery"] for row in rows if row["name"] == f"warm_plus_oac_seed{seed}")
        for seed in range(3)
    ])
    actor_search_recovery = np.asarray([
        next(row["strong_oracle_headroom_recovery"] for row in rows if row["name"] == f"search_from_oac_seed{seed}")
        for seed in range(3)
    ])
    analysis = {
        "qualification": "HIGHSPEED_STRONG_SEARCH_ORACLE_ANALYSIS_TRAIN_ONLY",
        "summary_sha256": sha256(summary_path), "validator_sha256": sha256(validator_path),
        "artifact_sha256": sha256(artifact_path), "contexts": 120,
        "baseline_rows": rows,
        "winning_start_counts": {name: int(np.sum(winning_start == index)) for index, name in enumerate(names)},
        "actor_start_to_terminal_residual_sigma_rms": paired_residual,
        "decision": {
            "stop_actor_optimization_if_guard_recovery_at_least": 0.80,
            "actor_guard_recovery_range": [float(actor_guard_recovery.min()), float(actor_guard_recovery.max())],
            "actor_seeded_search_recovery_range": [float(actor_search_recovery.min()), float(actor_search_recovery.max())],
            "qualification": (
                "ACTOR_GOOD_SEARCH_INITIALIZATION_MULTI_STEP_REFINEMENT_LIMIT"
                if actor_guard_recovery.max() < 0.80 and actor_search_recovery.min() >= 0.95
                else "REQUIRES_FURTHER_ROUTING"
            ),
            "next_teacher": "pair each OOF Actor center with its own search path; distill bounded intermediate refinements iteratively rather than the cross-basin global argmin in one step",
        },
        "limits": [
            "Train-only 120-state mechanism subset; not formal validation/test.",
            "Only control_step=0 is used to maximize independent episodes; later-step tail is not represented.",
            "Strong search is a numerical lower reference among 965 evaluated centers, not a proven global optimum.",
        ],
    }
    path = root / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis["decision"], indent=2))


if __name__ == "__main__":
    main()
