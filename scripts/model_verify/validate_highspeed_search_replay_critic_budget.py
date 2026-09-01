#!/usr/bin/env python3
"""Independently reload and validate high-speed search-Replay Critic budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import load_data
from train_highspeed_actor_visited_oac import build_inputs, local_critic_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-atol", type=float, default=2e-7)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_data(contract: dict) -> dict[str, np.ndarray]:
    root = Path(contract["source_pretrain"])
    summary = json.loads((root / "summary.json").read_text())
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
    report_path = root / "validator_report.json"
    summary = json.loads(summary_path.read_text())
    if summary["qualification"] != "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected training qualification")
    contract = summary["contract"]
    if not contract["actor_frozen"] or contract["new_rollout_count"] != 0:
        raise AssertionError("validator expects frozen-Actor, zero-rollout budget run")
    if contract["analytic_dbm_gradient"] or contract["formal_validation_or_test_created"]:
        raise AssertionError("forbidden contract flag")

    source_summary_path = Path(summary["sources"]["oac_summary"])
    source_validator_path = Path(summary["sources"]["oac_validator"])
    banks_path = Path(summary["sources"]["fresh_banks"])
    for path, key in (
        (source_summary_path, "oac_summary_sha256"),
        (source_validator_path, "oac_validator_sha256"),
        (banks_path, "fresh_banks_sha256"),
    ):
        if sha256(path) != summary["sources"][key]:
            raise AssertionError(f"source hash mismatch: {path}")
    source_validator = json.loads(source_validator_path.read_text())
    if source_validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("source OAC validator did not pass")
    source_root = source_summary_path.parent
    source_contract = json.loads((source_root / "contract.json").read_text())
    data = load_source_data(source_contract)
    source_summary = json.loads(source_summary_path.read_text())
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    with np.load(banks_path, allow_pickle=False) as loaded:
        banks = {name: np.asarray(loaded[name]) for name in loaded.files}
    bank_map = {
        (int(fold), int(seed)): index
        for index, (fold, seed) in enumerate(zip(banks["fold"], banks["seed"]))
    }

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    maximum_error = 0.0
    leakage_count = 0
    validated = []
    final_budget = max(int(value) for value in contract["extra_updates_per_twin"])
    for record in summary["records"]:
        key = (int(record["fold"]), int(record["seed"]))
        source_record = source_records[key]
        checkpoint = Path(record["checkpoint"])
        source_checkpoint = Path(record["source_checkpoint"])
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise AssertionError("Critic checkpoint hash mismatch")
        if sha256(source_checkpoint) != record["source_checkpoint_sha256"]:
            raise AssertionError("source checkpoint hash mismatch")
        if source_checkpoint != Path(source_record["checkpoint"]):
            raise AssertionError("source checkpoint path/key mismatch")
        payload = torch.load(checkpoint, map_location=device)
        source_payload = torch.load(source_checkpoint, map_location=device)
        if payload["formal_validation_or_test_created"]:
            raise AssertionError("checkpoint created forbidden split")
        if int(payload["extra_updates_per_twin"]) != final_budget:
            raise AssertionError("checkpoint budget mismatch")
        fit = np.asarray(payload["fit_indices"], np.int64)
        oof = np.asarray(payload["oof_indices"], np.int64)
        if np.intersect1d(data["episode"][fit], data["episode"][oof]).size:
            leakage_count += 1
        if not np.array_equal(fit, np.asarray(source_payload["fit_indices"], np.int64)):
            raise AssertionError("fit indices changed")
        if not np.array_equal(oof, np.asarray(source_payload["oof_indices"], np.int64)):
            raise AssertionError("OOF indices changed")
        inputs = build_inputs(data, source_payload)
        critics = []
        training = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            critics.append(critic)
            training.append(payload[f"critic{twin}_training"])
        replay_path = Path(source_record["replay"])
        if sha256(replay_path) != source_record["replay_sha256"]:
            raise AssertionError("source replay hash mismatch")
        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {name: np.asarray(loaded[name]) for name in loaded.files}
        bank_index = bank_map[key]
        if not np.array_equal(banks["oof"][bank_index], oof):
            raise AssertionError("fresh bank OOF mismatch")
        path_metric = local_critic_metrics(
            tuple(critics), tuple(training), inputs, oof,
            banks["bank"][bank_index], banks["cost"][bank_index], device,
        )
        local_metric = local_critic_metrics(
            tuple(critics), tuple(training), inputs, oof,
            replay["oof_probe_bank"], replay["oof_probe_cost"], device,
        )
        expected = next(
            row for row in record["curve"]
            if int(row["extra_updates_per_twin"]) == final_budget
        )
        error = max(
            metric_error(expected["path"], path_metric),
            metric_error(expected["local_0p05"], local_metric),
        )
        maximum_error = max(maximum_error, error)
        validated.append({"fold": key[0], "seed": key[1], "metric_max_abs_error": error})
    if leakage_count:
        raise AssertionError(f"episode leakage found in {leakage_count} runs")
    if maximum_error > args.metric_atol:
        raise AssertionError(
            f"metric replay error {maximum_error:.3e} exceeds {args.metric_atol:.3e}"
        )
    report = {
        "format": "highspeed_search_replay_critic_budget_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_INDEPENDENT_RELOAD_PASS",
        "contract": {
            "checkpoint_count": len(validated), "final_budget": final_budget,
            "actor_frozen": True, "new_rollout_count": 0,
            "analytic_dbm_gradient": False,
            "formal_validation_or_test_created": False,
        },
        "checks": {
            "summary_sha256": sha256(summary_path),
            "source_hashes_verified": True,
            "checkpoint_hashes_verified": True,
            "episode_leakage_count": leakage_count,
            "metric_max_abs_error": maximum_error,
            "metric_atol": args.metric_atol,
        },
        "records": validated,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
