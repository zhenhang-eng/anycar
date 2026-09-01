#!/usr/bin/env python3
"""Strict train-only output-support A/B for the differentiable DBM Actor.

All arms start from exactly the same K=1 G-X action.  A zero-initialized
adapter perturbs the pre-squash coordinate while only the support mapping
changes: current center+-1std, center+-3std, or full [-1,1].  The state set,
DBM J50 loss, optimizer, update count and random sampling are paired.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from run_mppi_absolute_action_value_critic_cv import make_folds
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_dbm_task_loss_actor import (
    DEFAULT_BANK, DEFAULT_ACTOR, DEFAULT_GT_V1, evaluate_model, load_actor,
    result_metrics, stratified_subset, train_task_loss,
)
from run_mppi_multi_candidate_actor import distribution


DEFAULT_OUTPUT = Path("outputs/mppi_proposal/dbm_task_loss_support_ab_20260825_v1")


class SupportExpandedActor(nn.Module):
    """Same G-X latent plan with a paired zero-start support adapter."""

    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__()
        if mode not in ("box1", "box3", "full"):
            raise ValueError(mode)
        self.base = copy.deepcopy(base)
        self.mode = mode
        self.adapter_head = nn.Linear(64, 2)
        self.adapter_skip = nn.Linear(192, 16)
        nn.init.zeros_(self.adapter_head.weight)
        nn.init.zeros_(self.adapter_head.bias)
        nn.init.zeros_(self.adapter_skip.weight)
        nn.init.zeros_(self.adapter_skip.bias)

    def forward(self, history, reference, current):
        feature, decoded = self.base.encode_latent(history, reference, current)
        raw = self.base.raw_from_latent(feature, decoded)[:, 0]
        delta = self.adapter_head(decoded) + self.adapter_skip(feature).reshape(-1, 8, 2)
        center = self.base.out_center[:, 0]
        scale = self.base.out_scale[:, 0]
        if self.mode == "box1":
            action = center + scale * torch.tanh(raw + delta)
        elif self.mode == "box3":
            base_unit = torch.tanh(raw)
            origin = torch.atanh(torch.clamp(base_unit / 3.0, -0.999999, 0.999999))
            action = center + 3.0 * scale * torch.tanh(origin + delta)
        else:
            base_action = torch.clamp(center + scale * torch.tanh(raw), -1.0, 1.0)
            origin = torch.atanh(torch.clamp(base_action, -0.999999, 0.999999))
            action = torch.tanh(origin + delta)
        return torch.clamp(action, -1.0, 1.0)[:, None]


def make_inputs(data, normalizer, device):
    values = normalizer.normalize_numpy(data["history"], data["reference"], data["current"])
    return tuple(torch.from_numpy(value.astype(np.float32)).to(device) for value in values)


def support_usage(action: np.ndarray, center: np.ndarray, scale: np.ndarray,
                  mode: str) -> dict:
    if mode == "box1":
        normalized = (action - center) / scale
    elif mode == "box3":
        normalized = (action - center) / (3.0 * scale)
    else:
        normalized = action
    absolute = np.abs(normalized)
    return {
        "component_ge_0_95": float(np.mean(absolute >= 0.95)),
        "state_any_ge_0_95": float(np.mean(np.any(absolute >= 0.95, axis=(1, 2)))),
        "steering_component_ge_0_95": float(np.mean(absolute[:, :, 1] >= 0.95)),
        "acceleration_component_ge_0_95": float(np.mean(absolute[:, :, 0] >= 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-count", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1200)
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
    train_pool = np.flatnonzero(folds != args.fold)
    subset = stratified_subset(data, train_pool, args.state_count, 260824 + args.state_count)
    checkpoint = args.actor_root / f"actor_k1_fold{args.fold}_seed{args.seed}.pt"
    base, payload = load_actor(checkpoint, device)
    normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
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
        reference_action, reference_initial = evaluate_model(
            base, inputs, backend, weights, params, states, current, reference,
            subset, device,
        )
    records = {}
    initial_actions = {}
    for mode in ("box1", "box3", "full"):
        torch.manual_seed(args.seed)
        model = SupportExpandedActor(base, mode).to(device)
        initial_action, initial_cost = evaluate_model(
            model, inputs, backend, weights, params, states, current, reference,
            subset, device,
        )
        initial_actions[mode] = initial_action
        initial_action_error = float(np.max(np.abs(initial_action - reference_action)))
        initial_cost_error = float(np.max(np.abs(initial_cost - reference_initial)))
        if initial_action_error > 2e-6 or initial_cost_error > 2e-3:
            raise AssertionError(
                f"{mode} initial mismatch: action={initial_action_error} "
                f"cost={initial_cost_error}"
            )
        checkpoint_path = args.output_dir / f"support_{mode}.pt"
        if args.resume_existing and checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu")
            if not np.array_equal(np.asarray(saved["subset_indices"]), subset):
                raise AssertionError(f"{mode} resume subset mismatch")
            final_action = np.asarray(saved["final_action"], np.float32)
            trace = saved["trace"]
            # Recompute cost instead of trusting the interrupted process.
            replay = torch.from_numpy(final_action).to(device)
            packed = torch.empty(len(states), 8, 2, device=device)
            packed[torch.from_numpy(subset).to(device)] = replay
            from run_mppi_dbm_task_loss_actor import rollout_cost
            final_cost = rollout_cost(
                backend, weights, params, packed, states, current, reference,
                subset, 128, device,
            ).cpu().numpy()
            trained = model
        else:
            trained, trace = train_task_loss(
                model, inputs, backend, weights, params, states, current, reference,
                subset, args.updates, args.batch_size, args.learning_rate,
                args.weight_decay, args.evaluation_interval, device,
                args.seed + args.state_count,
            )
            final_action, final_cost = evaluate_model(
                trained, inputs, backend, weights, params, states, current, reference,
                subset, device,
            )
        records[mode] = {
            "metrics": result_metrics(initial_cost, j16_cost[subset], final_cost),
            "initial_action_max_abs_error": initial_action_error,
            "initial_cost_max_abs_error": initial_cost_error,
            "support_usage": support_usage(final_action, center, scale, mode),
            "trace": trace,
        }
        if not checkpoint_path.exists():
            torch.save({
                "model_state_dict": trained.state_dict(), "mode": mode,
                "subset_indices": subset, "initial_action": initial_action,
                "final_action": final_action, "trace": trace,
            }, checkpoint_path)
        print(
            mode, records[mode]["metrics"]["final_cost"]["mean"],
            records[mode]["metrics"]["headroom_recovery"],
            records[mode]["support_usage"],
        )
    initial_action_error = max(
        float(np.max(np.abs(initial_actions[left] - initial_actions[right])))
        for left, right in (("box1", "box3"), ("box1", "full"))
    )
    baseline = records["box1"]["metrics"]
    best_mode = min(records, key=lambda key: records[key]["metrics"]["final_cost"]["mean"])
    best = records[best_mode]["metrics"]
    mean_improvement = baseline["final_cost"]["mean"] - best["final_cost"]["mean"]
    recovery_improvement = best["headroom_recovery"] - baseline["headroom_recovery"]
    support_pass = (
        best_mode != "box1" and mean_improvement >= 2.0
        and recovery_improvement >= 0.05
        and best["regressed_fraction"] <= baseline["regressed_fraction"] + 0.02
    )
    qualification = (
        "EXPANDED_ACTION_SUPPORT_MATERIAL_TINY128_GAIN"
        if support_pass else "EXPANDED_ACTION_SUPPORT_NO_MATERIAL_TINY128_GAIN"
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "contract": {
            "split": "train-only fold-0 stratified 128-state subset",
            "paired_initial_action_max_abs_error": initial_action_error,
            "loss": "deterministic DBM mean J50",
            "same_state_optimizer_updates_sampling": True,
            "critic_used": False, "teacher_used_for_training": False,
            "formal_validation_loaded": False, "test_loaded": False,
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "subset_indices": subset.tolist(),
        "j16_cost": distribution(j16_cost[subset]),
        "records": records,
        "decision": {
            "best_mode": best_mode,
            "mean_cost_improvement_vs_box1": mean_improvement,
            "recovery_improvement_vs_box1": recovery_improvement,
            "gate": "mean gain >=2, recovery gain >=0.05, regression <= box1+0.02",
            "full_train_ab_authorized": support_pass,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(qualification)


if __name__ == "__main__":
    main()
