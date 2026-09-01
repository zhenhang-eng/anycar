#!/usr/bin/env python3
"""Independent reload and DBM replay validation for the high-speed Actor K scan."""

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
from pretrain_highspeed_actor_twin_critic import actor_predict
from train_highspeed_actor_visited_oac import build_inputs, rollout_bank
from train_highspeed_critic_readiness_actor_transfer import load_shared_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--cost-atol-scale", type=float, default=2e-6)
    parser.add_argument("--center-atol", type=float, default=2e-5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    contract_path = root / "contract.json"
    summary_path = root / "summary.json"
    analysis_path = root / "analysis.json"
    contract = json.loads(contract_path.read_text())
    summary = json.loads(summary_path.read_text())
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
    if contract["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_CONTRACT_TRAIN_ONLY":
        raise AssertionError("unexpected contract qualification")
    if summary["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected summary qualification")
    if analysis is not None:
        allowed_analysis_qualifications = {
            "HIGHSPEED_FIXED_TRUST_K16_MARGINAL_BODY_GAIN_NO_TAIL_GAIN",
            "HIGHSPEED_90ROUND_BUDGET_MATTERS_K16_FASTER_ENDPOINT_SIMILAR",
        }
        if analysis["qualification"] not in allowed_analysis_qualifications:
            raise AssertionError("unexpected analysis qualification")
    if summary["contract_sha256"] != sha256(contract_path):
        raise AssertionError("contract hash mismatch")
    if analysis is not None and analysis["source"]["summary_sha256"] != sha256(summary_path):
        raise AssertionError("analysis/summary hash mismatch")
    if any((contract["formal_validation_or_test_created"], summary["formal_validation_or_test_created"])):
        raise AssertionError("forbidden split created")
    source_root = Path(contract["arguments"]["source_dir"]).resolve()
    budget_root = Path(contract["arguments"]["budget_dir"]).resolve()
    upstream = (
        (source_root / "summary.json", "source_summary_sha256"),
        (source_root / "validator_report.json", "source_validator_sha256"),
        (budget_root / "summary.json", "budget_summary_sha256"),
        (budget_root / "validator_report.json", "budget_validator_sha256"),
    )
    for path, key in upstream:
        if sha256(path) != contract[key]:
            raise AssertionError(f"upstream hash mismatch: {path}")
    data = load_shared_data(json.loads((source_root / "contract.json").read_text()))
    source_summary = json.loads((source_root / "summary.json").read_text())
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    max_center_error = 0.0
    max_center_cost_error = 0.0
    max_bank_cost_error = 0.0
    leakage_count = 0
    trust_violation_count = 0
    checkpoint_count = 0
    expected_pairs = set()
    for record in summary["records"]:
        fold, seed = int(record["fold"]), int(record["seed"])
        expected_pairs.add((fold, seed))
        initial_serialized = None
        for k in contract["k_values"]:
            arm = record["arms"][str(k)]
            checkpoint_path = Path(arm["checkpoint"])
            evaluation_path = Path(arm["evaluation"])
            if sha256(checkpoint_path) != arm["checkpoint_sha256"]:
                raise AssertionError("checkpoint hash mismatch")
            if sha256(evaluation_path) != arm["evaluation_sha256"]:
                raise AssertionError("evaluation hash mismatch")
            payload = torch.load(checkpoint_path, map_location=device)
            if payload["formal_validation_or_test_created"]:
                raise AssertionError("checkpoint used forbidden split")
            if (int(payload["k"]), int(payload["fold"]), int(payload["seed"])) != (k, fold, seed):
                raise AssertionError("checkpoint identity mismatch")
            source_row = source_records[(fold, seed)]
            if payload["source_oac_checkpoint_sha256"] != source_row["checkpoint_sha256"]:
                raise AssertionError("source Actor lineage mismatch")
            fit = np.asarray(payload["fit_indices"], np.int64)
            oof = np.asarray(payload["oof_indices"], np.int64)
            if np.intersect1d(data["episode"][fit], data["episode"][oof]).size:
                leakage_count += 1
            current_initial = json.dumps(arm["initial"], sort_keys=True)
            if initial_serialized is None:
                initial_serialized = current_initial
            elif current_initial != initial_serialized:
                raise AssertionError("K arms do not share the same initial metrics")
            for round_row in arm["rounds"]:
                target_trust = float(
                    contract["equal_budget"]["cumulative_actor_output_trust_sigma_rms"]
                )
                step_mode = contract["equal_budget"].get("round_step_mode", "exact")
                trust_tolerance = max(5e-5, 0.0021 * target_trust)
                projected = float(round_row["actor_projected_cumulative_step_sigma_rms"])
                raw = float(round_row["actor_raw_cumulative_step_sigma_rms"])
                factor = float(round_row["actor_cumulative_projection"])
                if step_mode == "exact":
                    violated = abs(projected - target_trust) > trust_tolerance
                elif step_mode == "cap_only":
                    violated = projected > target_trust + trust_tolerance
                    if raw <= target_trust and (
                        abs(projected - raw) > trust_tolerance
                        or abs(factor - 1.0) > 1e-8
                    ):
                        violated = True
                else:
                    raise AssertionError(f"unknown round step mode: {step_mode}")
                if violated:
                    trust_violation_count += 1
            with np.load(evaluation_path, allow_pickle=False) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
            if not np.array_equal(oof, arrays["oof_indices"]):
                raise AssertionError("OOF index mismatch")
            actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
            actor.load_state_dict(payload["actor_selected_state_dict"], strict=True)
            actor.eval()
            inputs = build_inputs(data, payload)
            center = actor_predict(actor, inputs, oof, device)
            max_center_error = max(
                max_center_error,
                float(np.max(np.abs(center - arrays["selected_oof_center"]))),
            )
            center_cost = rollout_bank(
                data, center[:, None], oof, backend, weights, params, device,
                args.rollout_batch_size,
            )[:, 0]
            max_center_cost_error = max(
                max_center_cost_error,
                float(np.max(np.abs(center_cost - arrays["selected_oof_cost"]))),
            )
            bank_cost = rollout_bank(
                data, arrays["local_bank"], oof, backend, weights, params, device,
                args.rollout_batch_size,
            )
            max_bank_cost_error = max(
                max_bank_cost_error,
                float(np.max(np.abs(bank_cost - arrays["local_cost"]))),
            )
            checkpoint_count += 1
    expected = {
        (fold, seed) for fold in contract["folds"] for seed in contract["seeds"]
    }
    if expected_pairs != expected:
        raise AssertionError("fold/seed coverage mismatch")
    if leakage_count or trust_violation_count:
        raise AssertionError("leakage or fixed-trust violation")
    if max_center_error > args.center_atol:
        raise AssertionError("Actor reload mismatch")
    tolerance = max(1e-3, args.cost_atol_scale * float(np.max(data["anchor_cost"])))
    if max(max_center_cost_error, max_bank_cost_error) > tolerance:
        raise AssertionError("DBM replay tolerance exceeded")
    report = {
        "format": "highspeed_actor_k_scan_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_K_SCAN_INDEPENDENT_RELOAD_REPLAY_PASS",
        "contract": {
            "checkpoint_count": checkpoint_count,
            "fold_seed_pair_count": len(expected_pairs),
            "episode_leakage_count": leakage_count,
            "fixed_trust_violation_count": trust_violation_count,
            "round_step_mode": contract["equal_budget"].get("round_step_mode", "exact"),
            "formal_validation_or_test_created": False,
        },
        "checks": {
            "contract_sha256": sha256(contract_path),
            "summary_sha256": sha256(summary_path),
            "analysis_sha256": sha256(analysis_path) if analysis_path.exists() else None,
            "upstream_hashes_verified": True,
            "initial_metrics_identical_across_K": True,
            "actor_reload_max_abs_error": max_center_error,
            "center_dbm_cost_max_abs_error": max_center_cost_error,
            "local_bank_dbm_cost_max_abs_error": max_bank_cost_error,
            "dbm_cost_tolerance": tolerance,
        },
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
