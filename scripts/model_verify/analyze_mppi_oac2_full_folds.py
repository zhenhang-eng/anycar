#!/usr/bin/env python3
"""Evaluate all OAC-2 selected Actors on their outer-heldout episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_oac2_continuous_actor import evaluate_actor
from train_mppi_online_absolute_sac import (
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
)
from validate_mppi_oac2_continuous_actor import load_actor_checkpoint


DEFAULT_RUNS = (
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold0_20260824_v1"),
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1"),
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold2_20260824_v1"),
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_full_3fold_20260824_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs=3, type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_gate(metrics: dict) -> dict[str, bool]:
    return {
        "mean_gain_positive": metrics["gain_vs_initial"]["mean"] > 0.0,
        "ci_lower_positive": metrics["gain_episode_bootstrap_ci95"][0] > 0.0,
        "median_gain_positive": metrics["gain_vs_initial"]["median"] > 0.0,
        "speed_2_4_nonnegative": metrics["by_speed"]["2.4"]["mean_gain"] >= 0.0,
        "speed_2_8_nonnegative": metrics["by_speed"]["2.8"]["mean_gain"] >= 0.0,
        "guard_p05_nonnegative": metrics["two_center_guard"]["gain_vs_warm"]["p05"] >= 0.0,
        "guard_worst_nonnegative": metrics["two_center_guard"]["gain_vs_warm"]["minimum"] >= 0.0,
        "not_saturated": metrics["state_any_saturation_fraction"] <= 0.10,
    }


def run_analysis(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    run_dirs = [Path(value) for value in args.run_dirs]
    contracts = [json.loads((path / "contract.json").read_text()) for path in run_dirs]
    first_arguments = contracts[0]["arguments"]
    bank_root = Path(first_arguments["bank_root"])
    gt_v1 = Path(first_arguments["gt_v1"])
    base_ac = Path(first_arguments["base_ac"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, _ = load_actor_normalization(base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    eval_args = SimpleNamespace(
        rollout_batch_size=int(first_arguments["rollout_batch_size"]),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    fold_records = []
    for run_dir, contract in sorted(
        zip(run_dirs, contracts), key=lambda pair: int(pair[1]["outer_fold"])
    ):
        fold = int(contract["outer_fold"])
        summary = json.loads((run_dir / "summary.json").read_text())
        validator = json.loads((run_dir / "validator_report.json").read_text())
        heldout = np.flatnonzero(folds == fold)
        heldout_episodes = sorted(np.unique(data["episode"][heldout]).tolist())
        if set(heldout_episodes) & set(contract["fit_episodes"]):
            raise AssertionError("outer-heldout episode leaked into OAC-2 fit")
        if set(heldout_episodes) & set(contract["internal_selection_episodes"]):
            raise AssertionError("outer-heldout episode leaked into internal selection")
        seed_records = []
        for source in summary["records"]:
            seed = int(source["seed"])
            actor_root = Path(contract["arguments"]["actor_root"])
            initial_actor, _ = load_actor_checkpoint(
                actor_root / f"a0_fold{fold}_seed{seed}.pt", device
            )
            selected_actor, _ = load_actor_checkpoint(
                run_dir / f"seed_{seed}" / "actor_selected.pt", device
            )
            initial_metrics, initial_cost = evaluate_actor(
                eval_args, initial_actor, actor_inputs, heldout, None, data,
                states, current, references, backend, weights, params, device,
                26082800 + fold * 100 + seed,
            )
            selected_metrics, _ = evaluate_actor(
                eval_args, selected_actor, actor_inputs, heldout, initial_cost,
                data, states, current, references, backend, weights, params,
                device, 26082900 + fold * 100 + seed,
            )
            gates = seed_gate(selected_metrics)
            seed_records.append({
                "seed": seed,
                "selected_round": int(source["selected_round"]),
                "initial_metrics": initial_metrics,
                "selected_metrics": selected_metrics,
                "gates": gates,
                "passed": bool(all(gates.values())),
            })
        passed_seed_count = sum(row["passed"] for row in seed_records)
        fold_records.append({
            "fold": fold,
            "run_dir": str(run_dir.resolve()),
            "contract_sha256": sha256_file(run_dir / "contract.json"),
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "validator_sha256": sha256_file(run_dir / "validator_report.json"),
            "source_validator_passed": bool(validator["passed"]),
            "heldout_state_count": int(len(heldout)),
            "heldout_episodes": heldout_episodes,
            "passed_seed_count": int(passed_seed_count),
            "passed": passed_seed_count >= 2,
            "seeds": seed_records,
        })
    passed_fold_count = sum(row["passed"] for row in fold_records)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC3_OUTER_HELDOUT_PASS_READY_FOR_SHORT_CLOSED_LOOP"
            if passed_fold_count >= 2
            else "OAC3_OUTER_HELDOUT_FAIL_ACTOR_NOT_READY_FOR_CLOSED_LOOP"
        ),
        "passed_fold_count": int(passed_fold_count),
        "required_fold_count": 2,
        "all_source_validators_passed": bool(all(
            row["source_validator_passed"] for row in fold_records
        )),
        "folds": fold_records,
        "new_dbm_rollouts": int(sum(
            2 * row["heldout_state_count"] * len(row["seeds"])
            for row in fold_records
        )),
        "formal_validation_loaded": False,
        "test_loaded": False,
        "limitations": [
            "outer-heldout episodes are internal train-pool CV, not formal validation/test",
            "direct Actor P05/worst remain visible even when the two-center guard passes",
            "this is fixed-state J_direct transfer, not vehicle-state closed-loop transfer",
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    result = run_analysis(args)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "passed_fold_count": result["passed_fold_count"],
        "folds": [{
            "fold": row["fold"],
            "passed_seed_count": row["passed_seed_count"],
            "mean_gain": [seed["selected_metrics"]["gain_vs_initial"]["mean"] for seed in row["seeds"]],
            "p05": [seed["selected_metrics"]["gain_vs_initial"]["p05"] for seed in row["seeds"]],
            "worst": [seed["selected_metrics"]["gain_vs_initial"]["minimum"] for seed in row["seeds"]],
        } for row in result["folds"]],
    }, indent=2))


if __name__ == "__main__":
    main()
