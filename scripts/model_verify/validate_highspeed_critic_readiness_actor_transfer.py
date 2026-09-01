#!/usr/bin/env python3
"""Independent reload/replay for the high-speed Critic-readiness transfer A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import actor_predict, load_data, sha256
from train_highspeed_actor_visited_oac import build_inputs, local_critic_metrics, rollout_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--center-atol", type=float, default=1e-4)
    parser.add_argument("--metric-atol", type=float, default=2e-7)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_data(source_root: Path) -> dict[str, np.ndarray]:
    contract = json.loads((source_root / "contract.json").read_text())
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


def metric_error(expected: dict, actual: dict) -> float:
    names = (
        "centered_log_cost_pearson", "center_relative_sign_accuracy",
        "bank_gain_recovery", "selected_beats_center_fraction",
    )
    return max(abs(float(expected[name]) - float(actual[name])) for name in names)


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    summary_path = root / "summary.json"
    contract_path = root / "contract.json"
    summary = json.loads(summary_path.read_text())
    contract = json.loads(contract_path.read_text())
    if file_sha256(contract_path) != summary["contract_sha256"]:
        raise AssertionError("contract hash mismatch")
    if summary["qualification"] != "HIGHSPEED_CRITIC_READINESS_ACTOR_TRANSFER_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected qualification")
    if contract["analytic_dbm_gradient"] or contract["formal_validation_or_test_created"]:
        raise AssertionError("forbidden contract flag")
    source_root = Path(contract["arguments"]["source_dir"]).resolve()
    budget_root = Path(contract["arguments"]["budget_dir"]).resolve()
    source_summary_path = source_root / "summary.json"
    source_validator_path = source_root / "validator_report.json"
    budget_summary_path = budget_root / "summary.json"
    budget_validator_path = budget_root / "validator_report.json"
    for path, key in (
        (source_summary_path, "source_summary_sha256"),
        (source_validator_path, "source_validator_sha256"),
        (budget_summary_path, "budget_summary_sha256"),
        (budget_validator_path, "budget_validator_sha256"),
    ):
        if file_sha256(path) != contract[key]:
            raise AssertionError(f"upstream hash mismatch: {path}")
    source_summary = json.loads(source_summary_path.read_text())
    budget_summary = json.loads(budget_summary_path.read_text())
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    budget_records = {
        (int(row["fold"]), int(row["seed"])): row for row in budget_summary["records"]
    }
    data = load_shared_data(source_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    max_center_error = 0.0
    max_saved_cost_error = 0.0
    max_metric_error = 0.0
    leakage_count = 0
    initial_pair_error = 0.0
    validated = []
    for record in summary["records"]:
        key = (int(record["fold"]), int(record["seed"]))
        source_record = source_records[key]
        source_checkpoint = Path(source_record["checkpoint"])
        source_hash = source_record["checkpoint_sha256"]
        if file_sha256(source_checkpoint) != source_hash:
            raise AssertionError("source checkpoint hash mismatch")
        for metric in ("teacher_gain_recovery", "beats_or_equals_warm_fraction"):
            left = record["arms"]["original"]["initial"]["oof"][metric]
            right = record["arms"]["absorbed1600"]["initial"]["oof"][metric]
            initial_pair_error = max(initial_pair_error, abs(float(left) - float(right)))
        for arm in ("original", "absorbed1600"):
            expected = record["arms"][arm]
            checkpoint = Path(expected["checkpoint"])
            replay_path = Path(expected["replay"])
            if file_sha256(checkpoint) != expected["checkpoint_sha256"]:
                raise AssertionError("checkpoint hash mismatch")
            if file_sha256(replay_path) != expected["replay_sha256"]:
                raise AssertionError("replay hash mismatch")
            payload = torch.load(checkpoint, map_location=device)
            if payload["formal_validation_or_test_created"]:
                raise AssertionError("checkpoint created forbidden split")
            if payload["source_oac_checkpoint_sha256"] != source_hash:
                raise AssertionError("source lineage mismatch")
            expected_critic_hash = (
                source_hash if arm == "original"
                else budget_records[key]["checkpoint_sha256"]
            )
            if payload["critic_initialization_sha256"] != expected_critic_hash:
                raise AssertionError("Critic arm initialization mismatch")
            fit = np.asarray(payload["fit_indices"], np.int64)
            oof = np.asarray(payload["oof_indices"], np.int64)
            if np.intersect1d(data["episode"][fit], data["episode"][oof]).size:
                leakage_count += 1
            inputs = build_inputs(data, payload)
            actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
            actor.load_state_dict(payload["actor_selected_state_dict"], strict=True)
            center = actor_predict(actor, inputs, oof, device)
            with np.load(replay_path, allow_pickle=False) as loaded:
                replay = {name: np.asarray(loaded[name]) for name in loaded.files}
            center_error = float(np.max(np.abs(center - replay["selected_oof_center"])))
            max_center_error = max(max_center_error, center_error)
            if center_error > args.center_atol:
                raise AssertionError(f"center reload error {center_error:.3e}")
            saved_cost = rollout_bank(
                data, replay["selected_oof_center"][:, None], oof, backend,
                weights, params, device, args.rollout_batch_size,
            )[:, 0]
            cost_error = float(np.max(np.abs(saved_cost - replay["selected_oof_cost"])))
            max_saved_cost_error = max(max_saved_cost_error, cost_error)
            tolerance = max(1e-3, 2e-6 * float(np.max(data["anchor_cost"][oof])))
            if cost_error > tolerance:
                raise AssertionError(f"saved-center DBM replay error {cost_error:.3e}")
            critics = []
            training = []
            for twin in (1, 2):
                critic = ConfigurableAbsoluteActionValueCritic().to(device)
                critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
                critics.append(critic)
                training.append(payload[f"critic{twin}_training"])
            path = local_critic_metrics(
                tuple(critics), tuple(training), inputs, oof,
                replay["path_bank"], replay["path_cost"], device,
            )
            local = local_critic_metrics(
                tuple(critics), tuple(training), inputs, oof,
                replay["local_bank"], replay["local_cost"], device,
            )
            error = max(
                metric_error(expected["critic"]["path"], path),
                metric_error(expected["critic"]["local_0p05"], local),
            )
            max_metric_error = max(max_metric_error, error)
            if error > args.metric_atol:
                raise AssertionError(f"Critic metric error {error:.3e}")
            validated.append({
                "fold": key[0], "seed": key[1], "arm": arm,
                "center_error": center_error, "saved_cost_error": cost_error,
                "critic_metric_error": error,
            })
    if leakage_count:
        raise AssertionError(f"episode leakage in {leakage_count} runs")
    if initial_pair_error != 0.0:
        raise AssertionError("paired arms did not start from identical Actor metrics")
    report = {
        "format": "highspeed_critic_readiness_actor_transfer_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_CRITIC_READINESS_ACTOR_TRANSFER_INDEPENDENT_REPLAY_PASS",
        "contract": {
            "validated_checkpoints": len(validated),
            "arms": ["original", "absorbed1600"],
            "analytic_dbm_gradient": False,
            "formal_validation_or_test_created": False,
        },
        "checks": {
            "summary_sha256": file_sha256(summary_path),
            "upstream_hashes_verified": True,
            "checkpoint_and_replay_hashes_verified": True,
            "episode_leakage_count": leakage_count,
            "initial_actor_pair_max_error": initial_pair_error,
            "selected_center_reload_max_error": max_center_error,
            "saved_center_dbm_replay_max_error": max_saved_cost_error,
            "critic_metric_max_error": max_metric_error,
        },
        "records": validated,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
