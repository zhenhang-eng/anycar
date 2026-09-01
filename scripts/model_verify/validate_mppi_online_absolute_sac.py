#!/usr/bin/env python3
"""Independently validate the OAC-0/OAC-1 absolute-action burn-in artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
)
from train_mppi_online_absolute_sac import (
    ROLE_NAMES,
    critic_state_inputs,
    final_metrics,
    load_bank,
    load_frozen_actor,
    module_digest,
    predict_actions,
    predict_flat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--bank-root", type=Path,
        default=Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"),
    )
    parser.add_argument(
        "--gt-v1", type=Path,
        default=Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1"),
    )
    parser.add_argument("--replay-check-count", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def max_metric_error(left: dict, right: dict) -> float:
    errors = []
    for key in ("bank_best_recall", "warm_false_stay"):
        errors.append(abs(float(left["flat_stay"][key]) - float(right["flat_stay"][key])))
    for key in ("material_pair_accuracy_conservative",):
        errors.append(abs(
            float(left["actor_visited"][key]) - float(right["actor_visited"][key])
        ))
    for key in ("lag2_final_accuracy", "initially_wrong_corrected_fraction"):
        errors.append(abs(
            float(left["bad_action_correction"][key])
            - float(right["bad_action_correction"][key])
        ))
    return max(errors, default=0.0)


def load_trained_critic(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def replay_subset(
    replay: dict[str, np.ndarray], count: int,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    device: torch.device,
) -> float:
    index = np.linspace(0, len(replay["cost"]) - 1, min(count, len(replay["cost"])), dtype=np.int64)
    state = replay["state_index"][index]
    knots = torch.from_numpy(replay["action"][index, None]).to(device)
    with torch.no_grad():
        cost = batched_cost(
            backend, weights, interpolate_knots(knots, params.horizon),
            torch.from_numpy(states[state]).to(device),
            torch.from_numpy(current[state]).to(device),
            torch.from_numpy(references[state]).to(device),
        )[:, 0].cpu().numpy()
    return float(np.max(np.abs(cost - replay["cost"][index])))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    outer_fold = int(contract["fold"])
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(dbm_json))
    )
    checks = {
        "contract_oac0": contract["qualification"] == "OAC0_CONTRACT_FROZEN",
        "outer_fold_registered": outer_fold in (0, 1, 2),
        "formal_validation_sealed": not contract["formal_validation_loaded"],
        "test_sealed": not contract["test_loaded"],
        "candidate_bank_hash": (
            sha256_file(args.bank_root / "candidate_bank.npz")
            == contract["source_hashes"]["candidate_bank"]
        ),
        "candidate_manifest_hash": (
            sha256_file(args.bank_root / "dataset_manifest.json")
            == contract["source_hashes"]["candidate_manifest"]
        ),
        "top_level_actor_update_zero": int(summary["actor_update_count"]) == 0,
    }
    records = []
    for source in summary["records"]:
        seed = int(source["seed"])
        seed_dir = args.run_dir / f"seed_{seed}"
        stored = json.loads((seed_dir / "summary.json").read_text())
        with np.load(seed_dir / "actor_visited_replay.npz", allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        critic1, payload1 = load_trained_critic(seed_dir / "critic1.pt", device)
        critic2, payload2 = load_trained_critic(seed_dir / "critic2.pt", device)
        flat_head, flat_payload = load_trained_critic(seed_dir / "flat_head.pt", device)
        inputs1 = critic_state_inputs(data, payload1)
        inputs2 = critic_state_inputs(data, payload2)
        flat_inputs = critic_state_inputs(data, {
            "training": {"normalization": flat_payload["normalization"]}
        })
        pred1 = predict_actions(
            critic1, inputs1, payload1, replay["state_index"], replay["action"], device
        )
        pred2 = predict_actions(
            critic2, inputs2, payload2, replay["state_index"], replay["action"], device
        )
        flat = predict_flat(
            flat_head, flat_inputs, replay["state_index"], replay["action"], device
        )
        best_index = np.argmin(data["costs"], axis=1)
        bank_best = data["actions"][np.arange(len(data["costs"])), best_index]
        flat_bank_best = predict_flat(
            flat_head, flat_inputs, np.arange(len(data["costs"])), bank_best, device
        )
        flat_warm = predict_flat(
            flat_head, flat_inputs, np.arange(len(data["costs"])),
            data["actions"][:, 0], device,
        )
        recomputed = final_metrics(
            data, replay, pred1, pred2, flat, flat_bank_best, flat_warm,
            float(contract["arguments"]["material_gap"]),
            float(contract["arguments"]["flat_gap"]),
        )
        actor, actor_path = load_frozen_actor(
            Path(contract["arguments"]["actor_root"]), outer_fold, seed, device
        )
        expected_rows = (
            int(contract["arguments"]["rounds"])
            * int(contract["arguments"]["contexts_per_round"])
            * len(ROLE_NAMES)
        )
        expected_per_role = (
            int(contract["arguments"]["rounds"])
            * int(contract["arguments"]["contexts_per_round"])
        )
        role_counts = {
            role: int(np.sum(replay["role"] == role)) for role in ROLE_NAMES
        }
        record_checks = {
            "actor_checkpoint_hash": (
                sha256_file(actor_path)
                == contract["actor_checkpoints"][str(seed)]["sha256"]
            ),
            "actor_module_hash_before_after_equal": (
                stored["actor"]["module_sha256_before"]
                == stored["actor"]["module_sha256_after"]
                == module_digest(actor)
            ),
            "actor_update_zero": (
                int(stored["actor"]["update_count"]) == 0
                and int(payload1["actor_update_count"]) == 0
                and int(payload2["actor_update_count"]) == 0
                and int(flat_payload["actor_update_count"]) == 0
            ),
            "replay_row_count": len(replay["cost"]) == expected_rows,
            "all_roles_complete": all(
                role_counts[role] == expected_per_role for role in ROLE_NAMES
            ),
            "replay_train_only": bool(
                np.all(folds[replay["state_index"]] != outer_fold)
            ),
            "saved_prediction_match": bool(
                np.max(np.abs(pred1 - replay["final_critic1"])) < 2e-5
                and np.max(np.abs(pred2 - replay["final_critic2"])) < 2e-5
                and np.max(np.abs(flat - replay["flat_probability"])) < 2e-5
            ),
            "metrics_match": max_metric_error(recomputed, stored["metrics"]) < 1e-7,
            "gates_match": recomputed["gates"] == stored["metrics"]["gates"],
            "qualification_match": (
                stored["qualification"]
                == ("OAC1_BURNIN_PASS" if recomputed["passed"] else "OAC1_BURNIN_FAIL")
            ),
        }
        dbm_error = replay_subset(
            replay, args.replay_check_count, backend, weights, params,
            states, current, references, device,
        )
        record_checks["dbm_replay_error_le_1e_4"] = dbm_error <= 1e-4
        checks[f"seed_{seed}"] = bool(all(record_checks.values()))
        records.append({
            "seed": seed, "checks": record_checks,
            "dbm_replay_max_abs_error": dbm_error,
            "metric_max_abs_error": max_metric_error(recomputed, stored["metrics"]),
            "recomputed_passed": recomputed["passed"],
        })
    passed_seed_count = sum(record["recomputed_passed"] for record in records)
    checks["top_level_passed_seed_count"] = (
        passed_seed_count == int(summary["passed_seed_count"])
    )
    passed = bool(all(checks.values()))
    report = {
        "qualification": (
            "OAC01_INDEPENDENT_VALIDATION_PASS" if passed
            else "OAC01_INDEPENDENT_VALIDATION_FAIL"
        ),
        "checks": checks,
        "records": records,
        "passed": passed,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.run_dir / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise AssertionError("OAC-0/OAC-1 independent validation failed")


if __name__ == "__main__":
    main()
