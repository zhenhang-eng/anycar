#!/usr/bin/env python3
"""Fresh two-stage bank evaluation for paired search-Replay OAC Critics."""

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
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import load_data
from train_highspeed_actor_visited_oac import (
    build_inputs,
    local_critic_metrics,
    make_two_stage_exploration_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--second-radius-ratio", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size), "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def load_shared_data(contract: dict) -> dict[str, np.ndarray]:
    pretrain_root = Path(contract["source_pretrain"])
    summary = json.loads((pretrain_root / "summary.json").read_text())
    source = summary["source"]
    data = load_data(Path(source["replay_dir"]), Path(source["teacher_dir"]))
    indices = np.asarray(summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    return {
        key: value[indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }


def actor_predict(
    model: DirectNoAnchorGTXActor, inputs: tuple[np.ndarray, ...],
    rows: np.ndarray, device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            local = rows[start : start + 64]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
            output.append(model(*tensors)[1].cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def evaluate_arm(
    root: Path, data: dict[str, np.ndarray], device: torch.device,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    summary_path = root / "summary.json"
    validator_path = root / "validator_report.json"
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    if validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError(f"source validator did not pass: {root}")
    records = []
    saved = {"bank": [], "cost": [], "fold": [], "seed": [], "oof": []}
    for expected in summary["records"]:
        checkpoint = Path(expected["checkpoint"])
        if sha256(checkpoint) != expected["checkpoint_sha256"]:
            raise AssertionError("checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location=device)
        oof = np.asarray(payload["oof_indices"], np.int64)
        inputs = build_inputs(data, payload)
        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
        actor.load_state_dict(payload["actor_selected_state_dict"], strict=True)
        center = actor_predict(actor, inputs, oof, device)
        bank, cost, diagnostics = make_two_stage_exploration_bank(
            data, center, oof, args.radius, args.second_radius_ratio,
            "search_recentered65", backend, weights, params, device,
            args.batch_size,
        )
        critics = []
        training = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            critics.append(critic)
            training.append(payload[f"critic{twin}_training"])
        metric = local_critic_metrics(
            tuple(critics), tuple(training), inputs, oof, bank, cost, device
        )
        first_best = cost[:, :33].min(axis=1)
        full_best = cost.min(axis=1)
        metric.update({
            "available_gain": distribution(cost[:, 0] - full_best),
            "stage2_incremental_gain": distribution(first_best - full_best),
            "stage2_strict_improvement_fraction": float(np.mean(
                first_best - full_best > 1e-5
            )),
            "bank_diagnostics": diagnostics,
        })
        records.append({
            "fold": int(payload["fold"]), "seed": int(payload["seed"]),
            "checkpoint_sha256": sha256(checkpoint), "metrics": metric,
        })
        saved["bank"].append(bank)
        saved["cost"].append(cost)
        saved["fold"].append(int(payload["fold"]))
        saved["seed"].append(int(payload["seed"]))
        saved["oof"].append(oof)
    arrays = {
        "bank": np.stack(saved["bank"]), "cost": np.stack(saved["cost"]),
        "fold": np.asarray(saved["fold"], np.int64),
        "seed": np.asarray(saved["seed"], np.int64),
        "oof": np.stack(saved["oof"]),
    }
    return records, arrays


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_dir.resolve()
    treatment_root = args.treatment_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    baseline_contract = json.loads((baseline_root / "contract.json").read_text())
    treatment_contract = json.loads((treatment_root / "contract.json").read_text())
    if baseline_contract["source_pretrain_summary_sha256"] != treatment_contract[
        "source_pretrain_summary_sha256"
    ]:
        raise AssertionError("A/B source data mismatch")
    data = load_shared_data(baseline_contract)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    baseline, baseline_arrays = evaluate_arm(
        baseline_root, data, device, backend, weights, params, args
    )
    treatment, treatment_arrays = evaluate_arm(
        treatment_root, data, device, backend, weights, params, args
    )
    baseline_map = {(row["fold"], row["seed"]): row for row in baseline}
    treatment_map = {(row["fold"], row["seed"]): row for row in treatment}
    if baseline_map.keys() != treatment_map.keys():
        raise AssertionError("A/B keys differ")
    metric_names = (
        "centered_log_cost_pearson", "center_relative_sign_accuracy",
        "bank_gain_recovery", "selected_beats_center_fraction",
        "stage2_strict_improvement_fraction",
    )
    metrics = {}
    for name in metric_names:
        left = [float(baseline_map[key]["metrics"][name]) for key in sorted(baseline_map)]
        right = [float(treatment_map[key]["metrics"][name]) for key in sorted(baseline_map)]
        metrics[name] = {
            "baseline": distribution(left), "treatment": distribution(right),
            "paired_delta_treatment_minus_baseline": distribution(
                np.asarray(right) - np.asarray(left)
            ),
        }
    output.mkdir(parents=True)
    baseline_npz = output / "baseline_banks.npz"
    treatment_npz = output / "treatment_banks.npz"
    np.savez_compressed(baseline_npz, **baseline_arrays)
    np.savez_compressed(treatment_npz, **treatment_arrays)
    analysis = {
        "format": "highspeed_search_replay_critic_ab_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "FRESH_TWO_STAGE_CRITIC_AB_COMPLETE_TRAIN_ONLY",
        "contract": {
            "bank": "center + 32 Hadamard at 1.0sigma + 32 rotated-Hadamard at 0.7sigma around incumbent",
            "analytic_dbm_gradient": False,
            "formal_validation_or_test_created": False,
        },
        "sources": {
            "baseline_summary_sha256": sha256(baseline_root / "summary.json"),
            "baseline_validator_sha256": sha256(baseline_root / "validator_report.json"),
            "treatment_summary_sha256": sha256(treatment_root / "summary.json"),
            "treatment_validator_sha256": sha256(treatment_root / "validator_report.json"),
            "baseline_banks_sha256": sha256(baseline_npz),
            "treatment_banks_sha256": sha256(treatment_npz),
        },
        "metrics": metrics,
        "baseline_records": baseline,
        "treatment_records": treatment,
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({"qualification": analysis["qualification"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
