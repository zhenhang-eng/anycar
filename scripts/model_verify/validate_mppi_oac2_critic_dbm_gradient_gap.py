#!/usr/bin/env python3
"""Independent replay validator for the OAC-2 Critic-vs-DBM gradient audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_online_absolute_sac import critic_state_inputs, load_bank
from train_mppi_oac2_continuous_actor import internal_split
from validate_mppi_oac2_continuous_actor import load_critic_checkpoint
from analyze_mppi_oac2_critic_dbm_gradient_gap import (
    critic_cost_gradient,
    evaluate_action_bank,
    exact_cost_gradient,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = json.loads((args.audit_dir / "analysis.json").read_text())
    run_dir = Path(analysis["contract"]["run_dir"])
    run_contract = json.loads((run_dir / "contract.json").read_text())
    run_args = run_contract["arguments"]
    data = load_bank(Path(run_args["bank_root"]))
    folds = make_folds(data, 3)
    _, selection, _, _ = internal_split(data, folds, int(run_contract["outer_fold"]))
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, Path(run_args["gt_v1"])
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    device = torch.device(args.device)
    with np.load(args.audit_dir / "audit.npz", allow_pickle=False) as loaded:
        artifact = {key: np.asarray(loaded[key]) for key in loaded.files}
    if not np.array_equal(artifact["selection_indices"], selection):
        raise AssertionError("selection index mismatch")
    rng = np.random.default_rng(260825900)
    local = np.sort(rng.choice(len(selection), size=args.sample_count, replace=False))
    indices = selection[local]
    records = []
    max_cost_error = max_gradient_error = max_value_error = max_step_cost_error = 0.0
    roles = analysis["contract"]["roles"]
    for row in analysis["records"]:
        seed = int(row["seed"])
        seed_dir = run_dir / f"seed_{seed}"
        critic1, payload1 = load_critic_checkpoint(seed_dir / "critic1.pt", device)
        critic2, payload2 = load_critic_checkpoint(seed_dir / "critic2.pt", device)
        inputs = critic_state_inputs(data, payload1)
        seed_record = {"seed": seed, "roles": {}}
        for role in roles:
            prefix = f"seed{seed}_{role}"
            actions = artifact[f"{prefix}_actions"][local]
            cost, gradient = exact_cost_gradient(
                actions, indices, states, current, reference, backend, weights,
                params, device, args.sample_count,
            )
            critic = critic_cost_gradient(
                critic1, payload1, critic2, payload2, inputs, actions, indices,
                device, args.sample_count,
            )
            cost_error = float(np.max(np.abs(cost - artifact[f"{prefix}_true_cost"][local])))
            gradient_error = float(np.max(np.abs(
                gradient - artifact[f"{prefix}_true_gradient_log"][local]
            )))
            value_error = max(
                float(np.max(np.abs(critic["q1"] - artifact[f"{prefix}_q1"][local]))),
                float(np.max(np.abs(critic["q2"] - artifact[f"{prefix}_q2"][local]))),
                float(np.max(np.abs(
                    critic["g_conservative"] - artifact[f"{prefix}_g_conservative"][local]
                ))),
            )
            max_cost_error = max(max_cost_error, cost_error)
            max_gradient_error = max(max_gradient_error, gradient_error)
            max_value_error = max(max_value_error, value_error)
            seed_record["roles"][role] = {
                "dbm_cost_max_abs_error": cost_error,
                "dbm_gradient_max_abs_error": gradient_error,
                "critic_value_gradient_max_abs_error": value_error,
            }
        actions = artifact[f"seed{seed}_parameter_step_actions"][local]
        step_cost = evaluate_action_bank(
            actions, indices, states, current, reference, backend, weights,
            params, device, args.sample_count,
        )
        step_error = float(np.max(np.abs(
            step_cost - artifact[f"seed{seed}_parameter_step_cost"][local]
        )))
        max_step_cost_error = max(max_step_cost_error, step_error)
        seed_record["parameter_step_dbm_cost_max_abs_error"] = step_error
        records.append(seed_record)
    checks = {
        "source_summary_hash": sha256_file(run_dir / "summary.json") == analysis["contract"]["run_summary_sha256"],
        "source_validator_hash": sha256_file(run_dir / "validator_report.json") == analysis["contract"]["run_validator_sha256"],
        "artifact_hash": sha256_file(args.audit_dir / "audit.npz") == analysis["artifact_sha256"],
        "source_metric_error_le_1e_3": float(analysis["dbm_source_cost_metric_max_abs_error"]) <= 1e-3,
        "dbm_cost_replay_le_1e_5": max_cost_error <= 1e-5,
        "dbm_gradient_replay_le_1e_5": max_gradient_error <= 1e-5,
        # Transformer/Conv CUDA kernels differ slightly when the validator
        # changes the forward batch from 64 to 48.  DBM quantities keep their
        # stricter gates; this tolerance is only for Critic numerical replay.
        "critic_replay_le_5e_4": max_value_error <= 5e-4,
        "parameter_step_cost_replay_le_1e_5": max_step_cost_error <= 1e-5,
        "formal_validation_sealed": not bool(analysis["formal_validation_loaded"]),
        "test_sealed": not bool(analysis["test_loaded"]),
        "networks_not_updated": not bool(analysis["contract"]["networks_updated"]),
    }
    passed = all(checks.values())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC2_CRITIC_DBM_GRADIENT_GAP_VALIDATION_PASS" if passed
            else "OAC2_CRITIC_DBM_GRADIENT_GAP_VALIDATION_FAIL"
        ),
        "passed": passed,
        "checks": checks,
        "sample_count_per_seed_role": int(args.sample_count),
        "max_errors": {
            "dbm_cost": max_cost_error,
            "dbm_gradient": max_gradient_error,
            "critic_value_or_gradient": max_value_error,
            "parameter_step_cost": max_step_cost_error,
        },
        "records": records,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.audit_dir / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
