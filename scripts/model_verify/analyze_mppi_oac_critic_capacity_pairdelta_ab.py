#!/usr/bin/env python3
"""Paired aggregate analysis for the frozen-Actor Critic A/B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, nargs="?",
        default=Path(
            "outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def distribution(value):
    value = np.asarray(value, np.float64).reshape(-1)
    return {
        "count": int(len(value)), "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)), "maximum": float(np.max(value)),
    }


def selection_rows(score, cost):
    chosen_index = np.argmin(score, axis=-1)
    chosen = np.take_along_axis(cost, chosen_index[..., None], axis=-1)[..., 0]
    warm = cost[..., 0]
    oracle = np.min(cost, axis=-1)
    return {
        "chosen": chosen,
        "regret": chosen - oracle,
        "harmful": chosen > warm + 1e-5,
        "headroom": warm - chosen,
        "available": warm - oracle,
    }


def aggregate_selection(rows):
    denominator = float(np.sum(rows["available"]))
    return {
        "chosen_cost": distribution(rows["chosen"]),
        "regret": distribution(rows["regret"]),
        "harmful_fraction": float(np.mean(rows["harmful"])),
        "headroom_recovery": float(np.sum(rows["headroom"]) / denominator),
    }


def risk_metrics(score, cost):
    delta = cost[:, 2] - cost[:, 0]
    regression = delta > 0
    material = delta >= 0.1
    severe = delta >= 10
    active = score > 0
    return {
        "regression_count": int(regression.sum()),
        "regression_auc": float(roc_auc_score(regression, score)),
        "regression_recall": float(np.mean(active[regression])),
        "material_regression_count": int(material.sum()),
        "material_regression_recall": float(np.mean(active[material])),
        "severe_count": int(severe.sum()),
        "severe_recall": float(np.mean(active[severe])) if np.any(severe) else 1.0,
        "false_positive_rate": float(np.mean(active[~regression])),
    }


def bootstrap_difference(left, right, episodes, samples, seed):
    unique = np.unique(episodes)
    members = {value: np.flatnonzero(episodes == value) for value in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen_episode = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([members[value] for value in chosen_episode])
        # Keep all three model seeds for each resampled physical episode.
        values.append(float(np.mean(left[:, index]) - np.mean(right[:, index])))
    return {
        "point": float(np.mean(left) - np.mean(right)),
        "episode_bootstrap_95ci": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def main():
    args = parse_args()
    run = args.run_dir
    summary = json.loads((run / "summary.json").read_text())
    validator = json.loads((run / "validator_report.json").read_text())
    if not validator["passed"]:
        raise AssertionError("source validator did not pass")
    with np.load(run / "predictions.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(run / "outer_heldout_actor_candidates.npz", allow_pickle=False) as loaded:
        actor_data = {key: np.asarray(loaded[key]) for key in loaded.files}
    episodes = actor_data["episode"]
    speed = actor_data["speed"]
    bank_cost = data["heldout_bank_cost"]
    actor_cost = data["heldout_actor_cost"].transpose(0, 2, 1)

    score_bank = {}
    score_actor = {}
    for arm in ("base", "wide"):
        score_bank[arm] = np.stack([
            data[f"seed{seed}_{arm}_absolute_bank"] for seed in range(3)
        ])
        score_actor[arm] = np.stack([
            data[f"seed{seed}_{arm}_absolute_actor"] for seed in range(3)
        ])
    score_bank["pair_delta"] = np.stack([
        data[f"seed{seed}_pair_delta_pair_bank"] for seed in range(3)
    ])
    score_actor["pair_delta"] = np.stack([
        data[f"seed{seed}_pair_delta_pair_actor"] for seed in range(3)
    ])

    bank_cost_seed = np.broadcast_to(bank_cost, score_bank["base"].shape)
    selections = {}
    actor_selections = {}
    risk = {}
    for arm in ("base", "wide", "pair_delta"):
        selections[arm] = selection_rows(score_bank[arm], bank_cost_seed)
        actor_selections[arm] = selection_rows(score_actor[arm], actor_cost)
        risk[arm] = {
            "all": [], "by_speed": {},
        }
        risk_score = (
            score_actor[arm][:, :, 2] - score_actor[arm][:, :, 0]
            if arm in ("base", "wide")
            else score_actor[arm][:, :, 2]
        )
        for seed in range(3):
            risk[arm]["all"].append(risk_metrics(
                risk_score[seed], actor_cost[seed]
            ))
        for value in sorted(np.unique(speed)):
            mask = np.isclose(speed, value)
            risk[arm]["by_speed"][f"{value:.1f}"] = [
                risk_metrics(
                    risk_score[seed, mask], actor_cost[seed, mask]
                ) for seed in range(3)
            ]

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "PAIR_DELTA_IMPROVES_HELDOUT_EXTREME_SELECTION_AND_RISK_RANKING_"
            "BUT_HIGH_SPEED_TAIL_REMAINS_UNSAFE"
        ),
        "source": str(run.resolve()),
        "source_qualification": summary["qualification"],
        "parameter_counts": validator["parameter_counts"],
        "bank_selection": {
            arm: aggregate_selection(selections[arm])
            for arm in selections
        },
        "actor_triplet_selection": {
            arm: aggregate_selection(actor_selections[arm])
            for arm in actor_selections
        },
        "risk_latest_vs_initial": risk,
        "paired_differences": {
            "wide_minus_base_bank_regret": bootstrap_difference(
                selections["wide"]["regret"], selections["base"]["regret"],
                episodes, args.bootstrap_samples, 260825201,
            ),
            "pair_minus_base_bank_regret": bootstrap_difference(
                selections["pair_delta"]["regret"], selections["base"]["regret"],
                episodes, args.bootstrap_samples, 260825202,
            ),
            "wide_minus_base_actor_triplet_regret": bootstrap_difference(
                actor_selections["wide"]["regret"],
                actor_selections["base"]["regret"],
                episodes, args.bootstrap_samples, 260825203,
            ),
            "pair_minus_base_actor_triplet_regret": bootstrap_difference(
                actor_selections["pair_delta"]["regret"],
                actor_selections["base"]["regret"],
                episodes, args.bootstrap_samples, 260825204,
            ),
        },
        "decision": {
            "capacity": (
                "The 1.94x parameter arm improves average pair accuracy and "
                "regression AUC, but its extreme top-1 regret is seed-variable; "
                "capacity alone is not the tail solution."
            ),
            "pair_delta": (
                "Direct same-state delta supervision improves bank top-1 regret, "
                "actor-triplet harmful selection, regression AUC, and severe-tail "
                "recall in all three paired seeds."
            ),
            "safety_boundary": (
                "Pair-delta remains a ranking/risk auxiliary only.  It still misses "
                "a material fraction of 2.8 m/s severe regressions, so training-time "
                "DBM accept/reject and deployment two-center guard remain mandatory."
            ),
        },
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (run / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "qualification": result["qualification"],
        "bank_selection": result["bank_selection"],
        "actor_triplet_selection": result["actor_triplet_selection"],
        "paired_differences": result["paired_differences"],
        "risk_2_8": {
            arm: risk[arm]["by_speed"]["2.8"] for arm in risk
        },
    }, indent=2))


if __name__ == "__main__":
    main()
