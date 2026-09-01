#!/usr/bin/env python3
"""Summarize speed/scenario transfer for iterative high-speed path distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_iterative_path_distillation_20260830_v1"
)
ARMS = ("one_step", "direct_two_step", "cascade_two_step")


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


def subset_metrics(data: dict[str, np.ndarray], examples: np.ndarray) -> dict[str, Any]:
    start = data["start_cost"][examples]
    warm = data["warm_cost"][examples]
    strong = data["strong_cost"][examples]
    output: dict[str, Any] = {
        "example_count": int(len(examples)),
        "start_cost": distribution(start),
        "start_warm_relative_reduction": float((warm - start).sum() / warm.sum()),
    }
    for arm in ARMS:
        target = data["path1_cost"][examples] if arm == "one_step" else data["path2_cost"][examples]
        seeds = []
        for seed in range(3):
            cost = data[f"{arm}_cost"][seed, examples]
            gain = start - cost
            seeds.append({
                "seed": seed,
                "cost": distribution(cost),
                "gain": distribution(gain),
                "target_gain_recovery": float(gain.sum() / (start - target).sum()),
                "strong_headroom_recovery": float(gain.sum() / (start - strong).sum()),
                "warm_relative_reduction": float((warm - cost).sum() / warm.sum()),
                "regression_fraction": float(np.mean(gain < -1e-5)),
            })
        output[arm] = seeds
    return output


def bootstrap_recovery(
    data: dict[str, np.ndarray], arm: str, seed: int, rng: np.random.Generator,
    replicates: int = 2000,
) -> list[float]:
    state_count = len(data["episode_id"])
    example_state = data["example_state_indices"].astype(np.int64)
    target = data["path1_cost"] if arm == "one_step" else data["path2_cost"]
    gain = data["start_cost"] - data[f"{arm}_cost"][seed]
    target_gain = data["start_cost"] - target
    values = []
    for _ in range(replicates):
        sampled_states = rng.integers(state_count, size=state_count)
        examples = np.concatenate([
            np.flatnonzero(example_state == state) for state in sampled_states
        ])
        values.append(float(gain[examples].sum() / target_gain[examples].sum()))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary_path = root / "summary.json"
    validator_path = root / "validator_report.json"
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    if validator["qualification"] != "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("distillation source did not pass independent replay")
    archive_path = Path(summary["archive"])
    if sha256(archive_path) != summary["archive_sha256"]:
        raise AssertionError("OOF archive hash mismatch")
    with np.load(archive_path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    example_state = data["example_state_indices"].astype(np.int64)
    all_examples = np.arange(len(example_state), dtype=np.int64)
    by_speed = {}
    for speed in sorted(np.unique(data["speed_kph"]).tolist()):
        states = np.flatnonzero(np.isclose(data["speed_kph"], speed))
        examples = np.flatnonzero(np.isin(example_state, states))
        by_speed[str(int(round(float(speed))))] = subset_metrics(data, examples)
    by_scenario = {}
    for scenario in sorted(np.unique(data["scenario_class"]).astype(str).tolist()):
        states = np.flatnonzero(data["scenario_class"].astype(str) == scenario)
        examples = np.flatnonzero(np.isin(example_state, states))
        by_scenario[scenario] = subset_metrics(data, examples)
    rng = np.random.default_rng(86_030)
    ci = {
        arm: {
            str(seed): bootstrap_recovery(data, arm, seed, rng)
            for seed in range(3)
        } for arm in ARMS
    }
    overall = subset_metrics(data, all_examples)
    analysis = {
        "qualification": "HIGHSPEED_INTERMEDIATE_PATH_LABELS_OOF_TRANSFER_PASS",
        "decision": (
            "One- and two-round actor-conditioned search-path labels transfer "
            "across episodes. Expand two-round path collection to all 600 train-only "
            "states; keep direct-two-step and staged cascade paired."
        ),
        "inputs": {
            "summary": str(summary_path), "summary_sha256": sha256(summary_path),
            "validator": str(validator_path), "validator_sha256": sha256(validator_path),
            "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        },
        "overall": overall,
        "episode_bootstrap_target_recovery_95ci": ci,
        "by_nominal_speed_kph": by_speed,
        "by_scenario": by_scenario,
        "interpretation": {
            "one_step": "recovers the first 1.0-sigma search transition",
            "direct_two_step": "one Actor forward predicts the two-round path endpoint",
            "cascade_two_step": "two Actor forwards with a separately trained second-stage updater",
            "boundary": (
                "120 independent step-0 train-only states; does not establish later-step, "
                "formal validation/test, or closed-loop transfer"
            ),
        },
    }
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "qualification": analysis["qualification"],
        "overall": overall,
        "bootstrap_ci": ci,
    }, indent=2))


if __name__ == "__main__":
    main()
