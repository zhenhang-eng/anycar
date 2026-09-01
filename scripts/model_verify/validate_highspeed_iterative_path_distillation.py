#!/usr/bin/env python3
"""Independently validate high-speed iterative path distillation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from pretrain_highspeed_actor_twin_critic import load_data
from train_highspeed_iterative_path_distillation import (
    ARMS,
    TRAINING_SEEDS,
    actor_forward,
    build_actor,
    evaluate_components,
)


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_iterative_path_distillation_20260830_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text())
    archive_path = Path(summary["archive"])
    if sha256(archive_path) != summary["archive_sha256"]:
        raise AssertionError("OOF archive hash mismatch")
    for key in ("strong_oracle", "strong_summary", "strong_validator", "replay", "teacher"):
        path = Path(summary["sources"][key])
        if sha256(path) != summary["sources"][f"{key}_sha256"]:
            raise AssertionError(f"source hash mismatch: {key}")
    if summary["contract"]["formal_validation_or_test_opened"]:
        raise AssertionError("artifact is not train-only")
    with np.load(archive_path, allow_pickle=False) as loaded:
        archive = {name: np.asarray(loaded[name]) for name in loaded.files}
    full = load_data(Path(summary["sources"]["replay"]).parent,
                     Path(summary["sources"]["teacher"]).parent)
    source_rows = archive["source_indices"].astype(np.int64)
    data = {
        key: (value[source_rows] if isinstance(value, np.ndarray) and value.ndim
              and len(value) == len(full["episode"]) else value)
        for key, value in full.items()
    }
    if not np.array_equal(data["episode"], archive["episode_id"]):
        raise AssertionError("replay/archive alignment failed")
    example_states = archive["example_state_indices"].astype(np.int64)
    folds = archive["fold"].astype(np.int64)
    predicted = {
        arm: np.full_like(archive[f"{arm}_centers"], np.nan) for arm in ARMS
    }
    leakage_violations = 0
    checkpoint_hash_violations = 0
    device = torch.device(args.device)
    records = {(int(row["fold"]), int(row["seed"])): row for row in summary["records"]}
    for fold in range(5):
        oof_examples = np.flatnonzero(folds[example_states] == fold)
        oof_states = example_states[oof_examples]
        for seed in TRAINING_SEEDS:
            record = records[(fold, seed)]
            sets = [set(record[name]) for name in (
                "fit_episodes", "selection_episodes", "oof_episodes"
            )]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                leakage_violations += 1
            models = {}
            inputs = None
            for arm in ("one_step", "direct_two_step", "stage2"):
                spec = record["training"][arm]
                checkpoint = Path(spec["checkpoint"])
                if sha256(checkpoint) != spec["checkpoint_sha256"]:
                    checkpoint_hash_violations += 1
                payload = torch.load(checkpoint, map_location="cpu")
                if (payload["arm"], int(payload["fold"]), int(payload["seed"])) != (arm, fold, seed):
                    raise AssertionError("checkpoint identity mismatch")
                actor = build_actor(float(payload["maximum_delta_sigma"]))
                actor.load_state_dict(payload["model_state_dict"], strict=True)
                actor.to(device); actor.eval(); models[arm] = actor
                normalizer = MPPIProposalNormalization.from_dict(payload["normalizer"])
                history, reference, current = normalizer.normalize_numpy(
                    data["history"], data["reference"], data["current"]
                )
                local_inputs = (
                    history.astype(np.float32), reference.astype(np.float32),
                    current.astype(np.float32), np.zeros((len(history), 74), np.float32),
                    np.zeros((len(history), 32), np.float32),
                )
                if inputs is None:
                    inputs = local_inputs
                elif any(not np.array_equal(a, b) for a, b in zip(inputs, local_inputs)):
                    raise AssertionError("normalizer differs between paired arms")
            assert inputs is not None
            start = archive["start_centers"][oof_examples]
            path1 = archive["path1_centers"][oof_examples]
            _, one = actor_forward(models["one_step"], inputs, oof_states, start, device)
            _, direct2 = actor_forward(
                models["direct_two_step"], inputs, oof_states, start, device
            )
            _, stage2 = actor_forward(models["stage2"], inputs, oof_states, path1, device)
            _, cascade = actor_forward(models["stage2"], inputs, oof_states, one, device)
            for arm, value in (
                ("one_step", one), ("direct_two_step", direct2),
                ("stage2", stage2), ("cascade_two_step", cascade),
            ):
                predicted[arm][seed, oof_examples] = value
    center_errors = {
        arm: float(np.max(np.abs(predicted[arm] - archive[f"{arm}_centers"])))
        for arm in ARMS
    }
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    cost_errors, component_errors = {}, {}
    for arm in ARMS:
        local_cost_errors, local_component_errors = [], []
        for seed in TRAINING_SEEDS:
            cost, parts = evaluate_components(
                data, example_states, predicted[arm][seed],
                backend, weights, params, device,
            )
            local_cost_errors.append(float(np.max(np.abs(
                cost - archive[f"{arm}_cost"][seed]
            ))))
            for key, value in parts.items():
                local_component_errors.append(float(np.max(np.abs(
                    value - archive[f"{arm}_{key}"][seed]
                ))))
        cost_errors[arm] = max(local_cost_errors)
        component_errors[arm] = max(local_component_errors)
    max_center = max(center_errors.values())
    max_cost = max(cost_errors.values())
    max_component = max(component_errors.values())
    passed = (
        leakage_violations == 0 and checkpoint_hash_violations == 0
        and max_center <= 2e-6 and max_cost <= 1.0 and max_component <= 1.0
    )
    qualification = (
        "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_INDEPENDENT_REPLAY_PASS"
        if passed else "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_INDEPENDENT_REPLAY_FAIL"
    )
    report = {
        "qualification": qualification,
        "leakage_violations": leakage_violations,
        "checkpoint_hash_violations": checkpoint_hash_violations,
        "center_max_abs_error": center_errors,
        "cost_max_abs_error": cost_errors,
        "component_max_abs_error": component_errors,
        "state_count": int(len(folds)), "example_count": int(len(example_states)),
        "checkpoint_count": int(5 * len(TRAINING_SEEDS) * 3),
        "formal_validation_or_test_opened": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
