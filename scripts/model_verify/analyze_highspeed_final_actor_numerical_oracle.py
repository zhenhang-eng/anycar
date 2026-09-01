#!/usr/bin/env python3
"""Consolidate the final high-speed Actor numerical headroom audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_highspeed_final_actor_numerical_oracle import distribution


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=26083197)
    return parser.parse_args()


def grouped(values: np.ndarray, keys: np.ndarray, actor: np.ndarray, best: np.ndarray) -> dict:
    result = {}
    for value in values:
        mask = keys == value
        gap = actor[mask] - best[mask]
        result[str(value)] = {
            "states": int(mask.sum()),
            "actor_mean_cost": float(actor[mask].mean()),
            "best_found_mean_cost": float(best[mask].mean()),
            "aggregate_gap_fraction_of_actor": float(gap.sum() / actor[mask].sum()),
            "paired_gap_fraction_median": float(np.median(gap / actor[mask])),
            "paired_gap_fraction_p95": float(np.quantile(gap / actor[mask], 0.95)),
        }
    return result


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir.resolve()
    with np.load(artifact / "solutions.npz", allow_pickle=False) as loaded:
        solution = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(artifact / "polish.npz", allow_pickle=False) as loaded:
        polish = {name: np.asarray(loaded[name]) for name in loaded.files}
    validator = json.loads((artifact / "validator_report.json").read_text())
    if validator["qualification"] != "HIGHSPEED_FINAL_ACTOR_NUMERICAL_ORACLE_REPLAY_PASS":
        raise AssertionError("numerical reference did not pass independent replay")
    warm = solution["initial_cost"][:, 0].astype(np.float64)
    old = solution["initial_cost"][:, 2].astype(np.float64)
    actor_seed = solution["initial_cost"][:, 3:6].astype(np.float64)
    actor = actor_seed.mean(axis=1)
    actor_best_seed = actor_seed.min(axis=1)
    best = polish["polished_cost"].astype(np.float64)
    gap = actor - best
    ratio = gap / actor
    total_headroom = warm - best
    actor_improvement = warm - actor

    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, len(actor), size=(args.bootstrap, len(actor)))
    boot_actor = actor[samples].sum(axis=1)
    boot_best = best[samples].sum(axis=1)
    boot_warm = warm[samples].sum(axis=1)
    boot_gap = (boot_actor - boot_best) / boot_actor
    boot_recovery = (boot_warm - boot_actor) / (boot_warm - boot_best)

    speed = solution["speed_kph"]
    scenario = solution["scenario_class"].astype(str)
    tail_indices = np.flatnonzero(ratio > 0.02)
    tail_records = [
        {
            "artifact_row": int(index),
            "source_index": int(solution["source_indices"][index]),
            "episode_id": str(solution["episode_id"][index]),
            "speed_kph": float(speed[index]),
            "scenario": str(scenario[index]),
            "fixed_actor_expected_cost": float(actor[index]),
            "numerical_best_found_cost": float(best[index]),
            "absolute_gap_J": float(gap[index]),
            "relative_gap_fraction": float(ratio[index]),
        }
        for index in tail_indices[np.argsort(ratio[tail_indices])[::-1]]
    ]
    analysis = {
        "format": "highspeed_final_actor_numerical_headroom_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "ACTOR_BODY_NEAR_NUMERICAL_CEILING_TAIL_TARGETED_HEADROOM_REMAINS",
        "interpretation_boundary": {
            "reference": "multi-start projected DBM autograd plus independent gradient-free polish",
            "certified_global_optimum": False,
            "proper_name": "numerical best-found reference",
            "objective": "deterministic DBM J50, physical [-1,1] action box, uniform 8 knots / 16D",
            "actor_metric": "three independently trained fixed Actor seeds averaged; no per-state seed oracle",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "headline": {
            "states": len(actor),
            "warm_mean_cost": float(warm.mean()),
            "old_oracle_mean_cost": float(old.mean()),
            "fixed_actor_three_seed_expected_mean_cost": float(actor.mean()),
            "numerical_best_found_mean_cost": float(best.mean()),
            "fixed_actor_aggregate_gap_fraction": float(gap.sum() / actor.sum()),
            "fixed_actor_aggregate_gap_fraction_bootstrap_ci95": np.quantile(
                boot_gap, (0.025, 0.975)
            ).tolist(),
            "warm_to_best_headroom_recovered_by_actor": float(
                actor_improvement.sum() / total_headroom.sum()
            ),
            "warm_to_best_headroom_recovery_bootstrap_ci95": np.quantile(
                boot_recovery, (0.025, 0.975)
            ).tolist(),
            "paired_gap_fraction": distribution(ratio),
            "absolute_gap_J": distribution(gap),
            "within_1pct_fraction": float(np.mean(ratio <= 0.01)),
            "within_2pct_fraction": float(np.mean(ratio <= 0.02)),
            "best_of_three_seed_aggregate_gap_fraction_non_deployable": float(
                (actor_best_seed - best).sum() / actor_best_seed.sum()
            ),
        },
        "per_fixed_actor_seed": validator["per_fixed_actor_seed"],
        "numerical_convergence": {
            "step": solution["trace_steps"].tolist(),
            "mean_best_cost": solution["trace_best_cost"].mean(axis=1).tolist(),
            "gradient_free_polish_aggregate_improvement_fraction": float(
                (solution["best_cost"] - best).sum() / solution["best_cost"].sum()
            ),
            "gradient_free_polish_states_improved": int(
                np.sum(solution["best_cost"] - best > 1e-4)
            ),
        },
        "by_speed_kph": grouped(np.unique(speed), speed, actor, best),
        "by_scenario": grouped(np.unique(scenario), scenario, actor, best),
        "tail_over_2pct": {
            "definition": "(fixed three-seed expected Actor J - best-found J) / Actor J > 0.02",
            "count": len(tail_records),
            "records": tail_records,
        },
    }
    (artifact / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
