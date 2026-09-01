#!/usr/bin/env python3
"""Full-600 cross-fit of one/two-stage high-speed Actor path distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from pretrain_highspeed_actor_twin_critic import load_data, make_folds
from train_highspeed_iterative_path_distillation import (
    ARMS,
    TRAINING_SEEDS,
    actor_forward,
    distribution,
    evaluate_components,
    normalize_inputs,
    train_actor,
)


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1"
)
DEFAULT_PATHS = Path(
    "outputs/mppi_proposal/highspeed_two_round_actor_paths_20260830_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_iterative_path_distillation_expansion_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32).reshape(1, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--paths-dir", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(
    cost: np.ndarray, start: np.ndarray, target: np.ndarray, warm: np.ndarray,
    parts: dict[str, np.ndarray], start_parts: dict[str, np.ndarray],
) -> dict[str, Any]:
    gain = start - cost
    total_gain = float(gain.sum())
    return {
        "cost": distribution(cost), "start_relative_gain": distribution(gain),
        "target_gain_recovery": float(gain.sum() / (start - target).sum()),
        "warm_relative_reduction": float((warm - cost).sum() / warm.sum()),
        "strictly_beats_start_fraction": float(np.mean(gain > 1e-5)),
        "regression_fraction": float(np.mean(gain < -1e-5)),
        "component_gain_fraction": {
            key: float((start_parts[key] - parts[key]).sum() / total_gain)
            for key in parts
        } if abs(total_gain) > 1e-12 else {},
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    path_artifact = (args.paths_dir / "paths.npz").resolve()
    path_summary_path = (args.paths_dir / "summary.json").resolve()
    path_validator_path = (args.paths_dir / "validator_report.json").resolve()
    path_summary = json.loads(path_summary_path.read_text())
    path_validator = json.loads(path_validator_path.read_text())
    if path_validator["qualification"] != "HIGHSPEED_TWO_ROUND_ACTOR_PATHS_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("path source did not pass independent replay")
    if path_summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("path source is not train-only")
    if sha256(path_artifact) != path_summary["artifact_sha256"]:
        raise AssertionError("path artifact hash mismatch")
    with np.load(path_artifact, allow_pickle=False) as loaded:
        paths = {name: np.asarray(loaded[name]) for name in loaded.files}
    data = load_data(args.replay_dir, args.teacher_dir)
    if not np.array_equal(data["episode"], paths["episode_id"]):
        raise AssertionError("path/replay alignment failed")
    path_centers = paths["path_centers"].astype(np.float32)
    path_costs = paths["path_costs"].astype(np.float32)
    state_count, source_seeds = path_centers.shape[:2]
    if path_centers.shape != (600, 3, 3, 8, 2):
        raise AssertionError("unexpected full path shape")
    example_states = np.repeat(np.arange(state_count, dtype=np.int64), source_seeds)
    start = path_centers[:, :, 0].reshape(-1, 8, 2)
    path1 = path_centers[:, :, 1].reshape(-1, 8, 2)
    path2 = path_centers[:, :, 2].reshape(-1, 8, 2)
    start_cost = path_costs[:, :, 0].reshape(-1)
    path1_cost = path_costs[:, :, 1].reshape(-1)
    path2_cost = path_costs[:, :, 2].reshape(-1)
    warm_cost = np.repeat(data["anchor_cost"], source_seeds)
    folds = make_folds(data["episode"], data["speed"], data["scenario"])
    if [int(np.sum(folds == fold)) for fold in range(5)] != [120] * 5:
        raise AssertionError("expected 120 physical contexts per fold")
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights(); params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    output.mkdir(parents=True)
    start_replay, start_parts = evaluate_components(
        data, example_states, start, backend, weights, params, device
    )
    label1_replay, label1_parts = evaluate_components(
        data, example_states, path1, backend, weights, params, device
    )
    label2_replay, label2_parts = evaluate_components(
        data, example_states, path2, backend, weights, params, device
    )
    if max(np.max(np.abs(start_replay - start_cost)),
           np.max(np.abs(label1_replay - path1_cost)),
           np.max(np.abs(label2_replay - path2_cost))) > 0.1:
        raise AssertionError("path replay mismatch")
    oof_centers = {
        arm: np.full((3, len(start), 8, 2), np.nan, np.float32) for arm in ARMS
    }
    records = []
    for fold in range(5):
        selection_fold = (fold + 1) % 5
        fit_states = np.flatnonzero((folds != fold) & (folds != selection_fold))
        selection_states = np.flatnonzero(folds == selection_fold)
        oof_states = np.flatnonzero(folds == fold)
        fit_examples = np.flatnonzero(np.isin(example_states, fit_states))
        selection_examples = np.flatnonzero(np.isin(example_states, selection_states))
        oof_examples = np.flatnonzero(np.isin(example_states, oof_states))
        inputs, normalizer = normalize_inputs(data, fit_states)
        for seed in TRAINING_SEEDS:
            models, training = {}, {}
            for arm, maximum, anchors, targets in (
                ("one_step", 1.0, start, path1),
                ("direct_two_step", 2.0, start, path2),
                ("stage2", 1.0, path1, path2),
            ):
                model, report, _ = train_actor(
                    arm, maximum, anchors, targets, example_states,
                    fit_examples, selection_examples, inputs, normalizer,
                    fold, seed, args, output,
                )
                models[arm] = model; training[arm] = report
            rows = example_states[oof_examples]
            _, one = actor_forward(models["one_step"], inputs, rows, start[oof_examples], device)
            _, direct2 = actor_forward(models["direct_two_step"], inputs, rows, start[oof_examples], device)
            _, stage2 = actor_forward(models["stage2"], inputs, rows, path1[oof_examples], device)
            _, cascade = actor_forward(models["stage2"], inputs, rows, one, device)
            for arm, value in (("one_step", one), ("direct_two_step", direct2),
                               ("stage2", stage2), ("cascade_two_step", cascade)):
                oof_centers[arm][seed, oof_examples] = value
            records.append({
                "fold": fold, "seed": seed,
                "fit_episodes": sorted(data["episode"][fit_states].astype(str).tolist()),
                "selection_episodes": sorted(data["episode"][selection_states].astype(str).tolist()),
                "oof_episodes": sorted(data["episode"][oof_states].astype(str).tolist()),
                "training": training,
            })
            print(json.dumps({"fold": fold, "seed": seed,
                "epochs": {key: value["best_epoch"] for key, value in training.items()},
                "selection_mse": {key: value["selection_mse"] for key, value in training.items()}}), flush=True)
    if any(not np.isfinite(value).all() for value in oof_centers.values()):
        raise AssertionError("incomplete OOF predictions")
    costs = {arm: np.empty((3, len(start)), np.float32) for arm in ARMS}
    parts_archive = {arm: {key: np.empty((3, len(start)), np.float32)
                           for key in start_parts} for arm in ARMS}
    per_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        report = {}
        for arm in ARMS:
            cost, parts = evaluate_components(
                data, example_states, oof_centers[arm][seed], backend, weights, params, device
            )
            costs[arm][seed] = cost
            for key in parts: parts_archive[arm][key][seed] = parts[key]
            target = path1_cost if arm == "one_step" else path2_cost
            report[arm] = metrics(cost, start_cost, target, warm_cost, parts, start_parts)
        transition_gain = path1_cost - costs["stage2"][seed]
        report["stage2"]["transition_gain_recovery"] = float(
            transition_gain.sum() / (path1_cost - path2_cost).sum()
        )
        per_seed[str(seed)] = report
    by_speed = {}
    for speed in sorted(np.unique(data["speed"]).tolist()):
        states = np.flatnonzero(np.isclose(data["speed"], speed))
        examples = np.flatnonzero(np.isin(example_states, states))
        by_speed[str(int(round(float(speed))))] = {
            arm: [metrics(
                costs[arm][seed, examples], start_cost[examples],
                (path1_cost if arm == "one_step" else path2_cost)[examples],
                warm_cost[examples],
                {key: value[seed, examples] for key, value in parts_archive[arm].items()},
                {key: value[examples] for key, value in start_parts.items()},
            ) for seed in TRAINING_SEEDS] for arm in ("one_step", "direct_two_step", "cascade_two_step")
        }
    gates = {}
    for arm in ("one_step", "direct_two_step", "cascade_two_step"):
        recovery = np.asarray([per_seed[str(seed)][arm]["target_gain_recovery"] for seed in TRAINING_SEEDS])
        p05 = np.asarray([per_seed[str(seed)][arm]["start_relative_gain"]["p05"] for seed in TRAINING_SEEDS])
        speed100_p05 = np.asarray([by_speed["100"][arm][seed]["start_relative_gain"]["p05"] for seed in TRAINING_SEEDS])
        position_fraction = np.asarray([
            per_seed[str(seed)][arm]["component_gain_fraction"]["position_along"]
            + per_seed[str(seed)][arm]["component_gain_fraction"]["position_cross"]
            for seed in TRAINING_SEEDS
        ])
        gates[arm] = {
            "target_gain_recovery": recovery.tolist(), "gain_p05": p05.tolist(),
            "speed100_gain_p05": speed100_p05.tolist(),
            "position_gain_fraction": position_fraction.tolist(),
            "at_least_2_of_3_recovery_ge_0_70": bool(np.sum(recovery >= 0.70) >= 2),
            "at_least_2_of_3_p05_nonnegative": bool(np.sum(p05 >= 0.0) >= 2),
            "at_least_2_of_3_speed100_p05_nonnegative": bool(np.sum(speed100_p05 >= 0.0) >= 2),
            "at_least_2_of_3_position_fraction_ge_0_40": bool(np.sum(position_fraction >= 0.40) >= 2),
        }
    archive_path = output / "oof_predictions.npz"
    archive: dict[str, np.ndarray] = {
        "source_indices": np.arange(state_count), "episode_id": data["episode"],
        "scenario_class": data["scenario"], "speed_kph": data["speed"], "fold": folds,
        "example_state_indices": example_states,
        "source_actor_seed": np.tile(np.arange(3, dtype=np.int64), state_count),
        "start_centers": start, "path1_centers": path1, "path2_centers": path2,
        "start_cost": start_cost, "path1_cost": path1_cost, "path2_cost": path2_cost,
        "warm_cost": warm_cost,
    }
    for arm in ARMS:
        archive[f"{arm}_centers"] = oof_centers[arm]; archive[f"{arm}_cost"] = costs[arm]
        for key, value in parts_archive[arm].items(): archive[f"{arm}_{key}"] = value
    np.savez_compressed(archive_path, **archive)
    label_metrics = {
        "path1": metrics(path1_cost, start_cost, path1_cost, warm_cost, label1_parts, start_parts),
        "path2": metrics(path2_cost, start_cost, path2_cost, warm_cost, label2_parts, start_parts),
    }
    summary = {
        "format": "highspeed_iterative_path_distillation_expansion_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_EXPANSION_PENDING_VALIDATION",
        "contract": {
            "physical_contexts": 600, "examples": 1800, "folds": 5,
            "fit_selection_oof_contexts": [360, 120, 120],
            "training_seeds": list(TRAINING_SEEDS),
            "formal_validation_or_test_opened": False,
        },
        "sources": {
            "paths": str(path_artifact), "paths_sha256": sha256(path_artifact),
            "paths_summary": str(path_summary_path), "paths_summary_sha256": sha256(path_summary_path),
            "paths_validator": str(path_validator_path), "paths_validator_sha256": sha256(path_validator_path),
            "replay": str((args.replay_dir / "replay.npz").resolve()), "replay_sha256": str(data["replay_sha256"]),
            "teacher": str((args.teacher_dir / "labels.npz").resolve()), "teacher_sha256": str(data["teacher_sha256"]),
        },
        "training": {"epochs": args.epochs, "patience": args.patience,
                     "batch_size": args.batch_size, "lr": args.lr},
        "label_metrics": label_metrics, "per_training_seed": per_seed,
        "by_nominal_speed_kph": by_speed, "gates": gates, "records": records,
        "archive": str(archive_path.resolve()), "archive_sha256": sha256(archive_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"label_metrics": label_metrics, "gates": gates,
                      "per_training_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()
