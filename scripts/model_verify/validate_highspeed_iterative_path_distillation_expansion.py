#!/usr/bin/env python3
"""Independent checkpoint/replay validation for full-600 path distillation."""

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
    ARMS, TRAINING_SEEDS, actor_forward, build_actor, evaluate_components,
)


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_iterative_path_distillation_expansion_20260830_v1"
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
        raise AssertionError("archive hash mismatch")
    for key in ("paths", "paths_summary", "paths_validator", "replay", "teacher"):
        if sha256(Path(summary["sources"][key])) != summary["sources"][f"{key}_sha256"]:
            raise AssertionError(f"source hash mismatch: {key}")
    if summary["contract"]["formal_validation_or_test_opened"]:
        raise AssertionError("artifact is not train-only")
    with np.load(archive_path, allow_pickle=False) as loaded:
        archive = {name: np.asarray(loaded[name]) for name in loaded.files}
    data = load_data(
        Path(summary["sources"]["replay"]).parent,
        Path(summary["sources"]["teacher"]).parent,
    )
    if not np.array_equal(data["episode"], archive["episode_id"]):
        raise AssertionError("archive/replay alignment mismatch")
    example_states = archive["example_state_indices"].astype(np.int64)
    folds = archive["fold"].astype(np.int64)
    predicted = {arm: np.full_like(archive[f"{arm}_centers"], np.nan) for arm in ARMS}
    records = {(int(row["fold"]), int(row["seed"])): row for row in summary["records"]}
    leakage = 0; hash_errors = 0
    device = torch.device(args.device)
    for fold in range(5):
        oof_examples = np.flatnonzero(folds[example_states] == fold)
        states = example_states[oof_examples]
        for seed in TRAINING_SEEDS:
            record = records[(fold, seed)]
            sets = [set(record[key]) for key in ("fit_episodes", "selection_episodes", "oof_episodes")]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                leakage += 1
            models = {}; inputs = None
            for arm in ("one_step", "direct_two_step", "stage2"):
                spec = record["training"][arm]; checkpoint = Path(spec["checkpoint"])
                if sha256(checkpoint) != spec["checkpoint_sha256"]: hash_errors += 1
                payload = torch.load(checkpoint, map_location="cpu")
                model = build_actor(float(payload["maximum_delta_sigma"]))
                model.load_state_dict(payload["model_state_dict"], strict=True)
                model.to(device); model.eval(); models[arm] = model
                normalizer = MPPIProposalNormalization.from_dict(payload["normalizer"])
                h, r, c = normalizer.normalize_numpy(
                    data["history"], data["reference"], data["current"]
                )
                local = (h.astype(np.float32), r.astype(np.float32), c.astype(np.float32),
                         np.zeros((600, 74), np.float32), np.zeros((600, 32), np.float32))
                if inputs is None: inputs = local
                elif any(not np.array_equal(a, b) for a, b in zip(inputs, local)):
                    raise AssertionError("paired normalizers differ")
            assert inputs is not None
            start = archive["start_centers"][oof_examples]
            path1 = archive["path1_centers"][oof_examples]
            _, one = actor_forward(models["one_step"], inputs, states, start, device)
            _, direct2 = actor_forward(models["direct_two_step"], inputs, states, start, device)
            _, stage2 = actor_forward(models["stage2"], inputs, states, path1, device)
            _, cascade = actor_forward(models["stage2"], inputs, states, one, device)
            for arm, value in (("one_step", one), ("direct_two_step", direct2),
                               ("stage2", stage2), ("cascade_two_step", cascade)):
                predicted[arm][seed, oof_examples] = value
    center_error = {arm: float(np.max(np.abs(
        predicted[arm] - archive[f"{arm}_centers"]
    ))) for arm in ARMS}
    backend = TorchDynamicBicycleRolloutBackend(); weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    cost_error, component_error = {}, {}
    for arm in ARMS:
        c_errors, p_errors = [], []
        for seed in TRAINING_SEEDS:
            cost, parts = evaluate_components(
                data, example_states, predicted[arm][seed], backend, weights, params, device
            )
            c_errors.append(float(np.max(np.abs(cost - archive[f"{arm}_cost"][seed]))))
            p_errors.extend(float(np.max(np.abs(
                value - archive[f"{arm}_{key}"][seed]
            ))) for key, value in parts.items())
        cost_error[arm] = max(c_errors); component_error[arm] = max(p_errors)
    passed = (leakage == 0 and hash_errors == 0 and max(center_error.values()) <= 2e-6
              and max(cost_error.values()) <= 1.0 and max(component_error.values()) <= 1.0)
    qualification = (
        "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_EXPANSION_INDEPENDENT_REPLAY_PASS"
        if passed else "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_EXPANSION_INDEPENDENT_REPLAY_FAIL"
    )
    report = {
        "qualification": qualification, "leakage_violations": leakage,
        "checkpoint_hash_violations": hash_errors,
        "center_max_abs_error": center_error, "cost_max_abs_error": cost_error,
        "component_max_abs_error": component_error,
        "physical_contexts": 600, "examples": 1800, "checkpoint_count": 45,
        "formal_validation_or_test_opened": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
