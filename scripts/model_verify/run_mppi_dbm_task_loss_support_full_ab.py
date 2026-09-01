#!/usr/bin/env python3
"""Full train/episode-heldout DBM task-loss A/B for Actor output support."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import make_folds
from run_mppi_dbm_task_loss_actor import (
    DEFAULT_ACTOR,
    DEFAULT_BANK,
    DEFAULT_GT_V1,
    evaluate_model,
    load_actor,
    result_metrics,
    rollout_cost,
    sha256_file,
    train_task_loss,
)
from run_mppi_dbm_task_loss_support_ab import SupportExpandedActor, make_inputs, support_usage


DEFAULT_OUTPUT = Path("outputs/mppi_proposal/dbm_task_loss_support_full_ab_20260825_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume_existing:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_existing)
    device = torch.device(args.device)

    with np.load(args.bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    folds = make_folds(data, 3)
    train_indices = np.flatnonzero(folds != args.fold)
    heldout_indices = np.flatnonzero(folds == args.fold)
    checkpoint = args.actor_root / f"actor_k1_fold{args.fold}_seed{args.seed}.pt"
    base, actor_payload = load_actor(checkpoint, device)
    normalizer = MPPIProposalNormalization.from_dict(actor_payload["normalization"])
    inputs = make_inputs(data, normalizer, device)
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, args.gt_v1
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    best_index = np.argmin(data["costs"][:, 16:24], axis=1)
    j16_cost = data["costs"][:, 16:24][np.arange(len(data["costs"])), best_index]
    center = base.out_center.detach().cpu().numpy().reshape(8, 2)
    scale = base.out_scale.detach().cpu().numpy().reshape(8, 2)

    with torch.no_grad():
        reference_train_action, reference_train_cost = evaluate_model(
            base, inputs, backend, weights, params, states, current, reference,
            train_indices, device,
        )
        reference_heldout_action, reference_heldout_cost = evaluate_model(
            base, inputs, backend, weights, params, states, current, reference,
            heldout_indices, device,
        )

    records = {}
    initial_actions = {}
    for mode in ("box1", "box3"):
        torch.manual_seed(args.seed)
        model = SupportExpandedActor(base, mode).to(device)
        initial_train_action, initial_train_cost = evaluate_model(
            model, inputs, backend, weights, params, states, current, reference,
            train_indices, device,
        )
        initial_heldout_action, initial_heldout_cost = evaluate_model(
            model, inputs, backend, weights, params, states, current, reference,
            heldout_indices, device,
        )
        initial_actions[mode] = np.concatenate(
            [initial_train_action, initial_heldout_action], axis=0
        )
        initial_action_error = max(
            float(np.max(np.abs(initial_train_action - reference_train_action))),
            float(np.max(np.abs(initial_heldout_action - reference_heldout_action))),
        )
        initial_cost_error = max(
            float(np.max(np.abs(initial_train_cost - reference_train_cost))),
            float(np.max(np.abs(initial_heldout_cost - reference_heldout_cost))),
        )
        if initial_action_error > 2e-6 or initial_cost_error > 2e-3:
            raise AssertionError(
                f"{mode} initial mismatch action={initial_action_error} cost={initial_cost_error}"
            )

        checkpoint_path = args.output_dir / f"support_{mode}.pt"
        if args.resume_existing and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu")
            if not np.array_equal(np.asarray(payload["train_indices"]), train_indices):
                raise AssertionError(f"{mode} resume train split mismatch")
            if not np.array_equal(np.asarray(payload["heldout_indices"]), heldout_indices):
                raise AssertionError(f"{mode} resume heldout split mismatch")
            trained = model
            trace = payload["trace"]
            train_action = np.asarray(payload["train_action"], dtype=np.float32)
            heldout_action = np.asarray(payload["heldout_action"], dtype=np.float32)
            packed = torch.zeros(len(states), 8, 2, dtype=torch.float32, device=device)
            packed[torch.from_numpy(train_indices).to(device)] = torch.from_numpy(train_action).to(device)
            packed[torch.from_numpy(heldout_indices).to(device)] = torch.from_numpy(heldout_action).to(device)
            train_cost = rollout_cost(
                backend, weights, params, packed, states, current, reference,
                train_indices, 128, device,
            ).cpu().numpy()
            heldout_cost = rollout_cost(
                backend, weights, params, packed, states, current, reference,
                heldout_indices, 128, device,
            ).cpu().numpy()
        else:
            trained, trace = train_task_loss(
                model, inputs, backend, weights, params, states, current, reference,
                train_indices, args.updates, args.batch_size, args.learning_rate,
                args.weight_decay, args.evaluation_interval, device, args.seed + 1000,
            )
            train_action, train_cost = evaluate_model(
                trained, inputs, backend, weights, params, states, current, reference,
                train_indices, device,
            )
            heldout_action, heldout_cost = evaluate_model(
                trained, inputs, backend, weights, params, states, current, reference,
                heldout_indices, device,
            )
            torch.save({
                "model_state_dict": trained.state_dict(),
                "mode": mode,
                "fold": args.fold,
                "seed": args.seed,
                "train_indices": train_indices,
                "heldout_indices": heldout_indices,
                "initial_train_action": initial_train_action,
                "initial_heldout_action": initial_heldout_action,
                "train_action": train_action,
                "heldout_action": heldout_action,
                "trace": trace,
            }, checkpoint_path)

        records[mode] = {
            "train": result_metrics(initial_train_cost, j16_cost[train_indices], train_cost),
            "episode_heldout": result_metrics(
                initial_heldout_cost, j16_cost[heldout_indices], heldout_cost
            ),
            "train_support_usage": support_usage(train_action, center, scale, mode),
            "heldout_support_usage": support_usage(heldout_action, center, scale, mode),
            "initial_action_max_abs_error": initial_action_error,
            "initial_cost_max_abs_error": initial_cost_error,
            "trace": trace,
        }
        print(
            mode,
            "train", records[mode]["train"]["final_cost"]["mean"],
            "heldout", records[mode]["episode_heldout"]["final_cost"]["mean"],
            flush=True,
        )

    paired_initial_error = float(np.max(np.abs(initial_actions["box1"] - initial_actions["box3"])))
    comparisons = {}
    conditions = []
    for split in ("train", "episode_heldout"):
        base_metrics = records["box1"][split]
        wide_metrics = records["box3"][split]
        mean_improvement = (
            base_metrics["final_cost"]["mean"] - wide_metrics["final_cost"]["mean"]
        )
        recovery_improvement = (
            wide_metrics["headroom_recovery"] - base_metrics["headroom_recovery"]
        )
        p05_gain_delta = wide_metrics["gain"]["p05"] - base_metrics["gain"]["p05"]
        regression_delta = (
            wide_metrics["regressed_fraction"] - base_metrics["regressed_fraction"]
        )
        comparisons[split] = {
            "mean_cost_improvement": mean_improvement,
            "recovery_improvement": recovery_improvement,
            "p05_gain_delta": p05_gain_delta,
            "regressed_fraction_delta": regression_delta,
        }
        conditions.extend([
            mean_improvement >= 2.0,
            recovery_improvement >= 0.05,
            p05_gain_delta >= -0.5,
            regression_delta <= 0.02,
        ])
    passed = all(conditions)
    qualification = (
        "BOX3_SUPPORT_FULL_TRAIN_AND_HELDOUT_PASS_READY_FOR_OAC_INTEGRATION"
        if passed else "BOX3_SUPPORT_FULL_REPLICATION_FAIL_REMAIN_DIAGNOSTIC_ONLY"
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "contract": {
            "split": "episode-grouped fold-0 train-1200 / heldout-600",
            "paired_initial_action_max_abs_error": paired_initial_error,
            "loss": "deterministic DBM mean J50",
            "same_state_optimizer_updates_sampling": True,
            "critic_used": False,
            "teacher_used_for_training": False,
            "j16_used_only_for_evaluation": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {
            "actor_checkpoint": str(checkpoint.resolve()),
            "actor_checkpoint_sha256": sha256_file(checkpoint),
            "candidate_bank": str((args.bank_root / "candidate_bank.npz").resolve()),
            "candidate_bank_sha256": sha256_file(args.bank_root / "candidate_bank.npz"),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_indices": train_indices.tolist(),
        "heldout_indices": heldout_indices.tolist(),
        "records": records,
        "comparison": comparisons,
        "decision": {
            "gate": (
                "both train and heldout: mean J improvement >=2; recovery gain >=0.05; "
                "gain P05 delta >=-0.5; regression fraction delta <=0.02"
            ),
            "oac_box3_integration_authorized": passed,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(qualification)


if __name__ == "__main__":
    main()
