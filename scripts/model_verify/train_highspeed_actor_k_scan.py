#!/usr/bin/env python3
"""Scan Actor microsteps K under a fixed cumulative per-round output trust."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import actor_predict, distribution, sha256
from train_highspeed_actor_visited_oac import (
    SIGMA,
    actor_metrics,
    actor_update,
    build_inputs,
    critic_update,
    evaluate_actor,
    make_exploration_bank,
    make_two_stage_exploration_bank,
    rollout_bank,
)
from train_highspeed_critic_readiness_actor_transfer import load_shared_data


DEFAULT_SOURCE = Path("outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1")
DEFAULT_BUDGET = Path("outputs/mppi_proposal/highspeed_search_replay_critic_budget_20260830_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--budget-dir", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="1,4,8,16")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--evaluation-rounds", default="",
        help="Comma-separated outer rounds for additional OOF learning-curve evaluation.",
    )
    parser.add_argument("--critic-updates-per-round", type=int, default=20)
    parser.add_argument("--critic-state-batch-size", type=int, default=8)
    parser.add_argument("--critic-candidates-per-state", type=int, default=48)
    parser.add_argument("--actor-batch-size", type=int, default=64)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument(
        "--actor-lr-decay-start-round", type=int, default=0,
        help="Start round for cosine Actor LR decay; zero disables decay.",
    )
    parser.add_argument(
        "--actor-lr-final-scale", type=float, default=1.0,
        help="Final Actor LR divided by --actor-lr when cosine decay is enabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.08)
    parser.add_argument("--exploration-radius-start", type=float, default=0.20)
    parser.add_argument("--exploration-radius-end", type=float, default=0.05)
    parser.add_argument(
        "--exploration-decay-rounds", type=int, default=0,
        help=(
            "Rounds used to anneal exploration radius; zero uses --rounds. "
            "Later rounds keep the end radius."
        ),
    )
    parser.add_argument("--second-radius-ratio", type=float, default=0.70)
    parser.add_argument("--max-round-step-sigma-rms", type=float, default=0.02)
    parser.add_argument(
        "--round-step-mode", choices=("exact", "cap_only"), default="exact",
        help=(
            "exact rescales every round to the requested output step; cap_only "
            "only clips updates that exceed the requested output step"
        ),
    )
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--raw-cost-weight-cap", type=float, default=8.0)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def actor_lr_for_round(args: argparse.Namespace, round_index: int) -> float:
    start = int(args.actor_lr_decay_start_round)
    final_scale = float(args.actor_lr_final_scale)
    if start <= 0 or round_index <= start:
        return float(args.actor_lr)
    if not 0.0 < final_scale <= 1.0:
        raise ValueError("--actor-lr-final-scale must be in (0, 1]")
    progress = min(1.0, (round_index - start) / max(args.rounds - start, 1))
    scale = final_scale + (1.0 - final_scale) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return float(args.actor_lr) * scale


def load_critic(payload: dict, twin: int, device: torch.device):
    model = ConfigurableAbsoluteActionValueCritic().to(device)
    model.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
    return model


def project_round_parameters(
    actor: DirectNoAnchorGTXActor, before: dict[str, torch.Tensor],
    before_action: np.ndarray, inputs, fit: np.ndarray,
    limit: float, mode: str, device: torch.device,
) -> tuple[float, float, float]:
    raw_action = actor_predict(actor, inputs, fit, device)
    raw = float(np.sqrt(np.mean(((raw_action - before_action) / SIGMA) ** 2)))
    after = {
        name: value.detach().clone() for name, value in actor.named_parameters()
    }
    if mode == "cap_only" and raw <= limit:
        return raw, raw, 1.0
    if mode not in {"exact", "cap_only"}:
        raise ValueError(f"unknown round step mode: {mode}")
    factor = limit / max(raw, 1e-12)
    projected = raw
    for _ in range(5):
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                parameter.copy_(before[name] + factor * (after[name] - before[name]))
        projected_action = actor_predict(actor, inputs, fit, device)
        projected = float(np.sqrt(np.mean(((projected_action - before_action) / SIGMA) ** 2)))
        if abs(projected / limit - 1.0) <= 0.002:
            break
        factor *= limit / max(projected, 1e-12)
    return raw, projected, factor


def aggregate(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    rows = [record["arms"][str(k)] for record in records]
    selected = [row["selected"]["oof"] for row in rows]
    return {
        "selected_round": distribution([row["selected_round"] for row in rows]),
        "teacher_gain_recovery": distribution([
            row["teacher_gain_recovery"] for row in selected
        ]),
        "gain_vs_start_mean": distribution([
            row["gain_vs_pretrained_actor"]["mean"] for row in selected
        ]),
        "gain_vs_start_p05": distribution([
            row["gain_vs_pretrained_actor"]["p05"] for row in selected
        ]),
        "gain_vs_warm_mean": distribution([
            row["gain_vs_warm"]["mean"] for row in selected
        ]),
        "gain_vs_warm_p05": distribution([
            row["gain_vs_warm"]["p05"] for row in selected
        ]),
        "beats_warm_fraction": distribution([
            row["beats_or_equals_warm_fraction"] for row in selected
        ]),
        "latest_minus_selected_mean_gain": distribution([
            row["latest"]["oof"]["gain_vs_pretrained_actor"]["mean"]
            - row["selected"]["oof"]["gain_vs_pretrained_actor"]["mean"]
            for row in rows
        ]),
    }


def run_arm(
    k: int, fold: int, seed: int, args: argparse.Namespace,
    data: dict[str, np.ndarray], source_record: dict, budget_record: dict,
    output: Path, device: torch.device,
) -> dict[str, Any]:
    evaluation_rounds = {
        int(value) for value in args.evaluation_rounds.split(",") if value.strip()
    }
    source_path = Path(source_record["checkpoint"])
    replay_path = Path(source_record["replay"])
    budget_path = Path(budget_record["checkpoint"])
    if sha256(source_path) != source_record["checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    if sha256(replay_path) != source_record["replay_sha256"]:
        raise AssertionError("source replay hash mismatch")
    if sha256(budget_path) != budget_record["checkpoint_sha256"]:
        raise AssertionError("budget checkpoint hash mismatch")
    source = torch.load(source_path, map_location=device)
    budget = torch.load(budget_path, map_location=device)
    if budget["source_checkpoint_sha256"] != source_record["checkpoint_sha256"]:
        raise AssertionError("budget/source lineage mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        replay_actions = np.asarray(loaded["actions"]).copy()
        replay_costs = np.asarray(loaded["costs"]).copy()
    fit = np.asarray(source["fit_indices"], np.int64)
    selection = np.asarray(source["selection_indices"], np.int64)
    oof = np.asarray(source["oof_indices"], np.int64)
    if np.intersect1d(data["episode"][fit], data["episode"][oof]).size:
        raise AssertionError("fit/OOF episode leakage")
    inputs = build_inputs(data, source)
    actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    actor.load_state_dict(source["actor_selected_state_dict"], strict=True)
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay
    )
    actor_optimizer.load_state_dict(source["actor_optimizer"])
    for group in actor_optimizer.param_groups:
        group["lr"] = args.actor_lr
        group["weight_decay"] = args.weight_decay
    critics = tuple(load_critic(budget, twin, device) for twin in (1, 2))
    training = tuple(budget[f"critic{twin}_training"] for twin in (1, 2))
    critic_optimizers = []
    for twin, critic in zip((1, 2), critics):
        optimizer = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(budget[f"critic{twin}_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.critic_lr
            group["weight_decay"] = args.weight_decay
        critic_optimizers.append(optimizer)
    critic_optimizers = tuple(critic_optimizers)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    initial = {}
    initial_cost = {}
    for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
        center = actor_predict(actor, inputs, rows, device)
        cost = rollout_bank(
            data, center[:, None], rows, backend, weights, params, device,
            args.rollout_batch_size,
        )[:, 0]
        initial[name] = actor_metrics(cost, rows, data, cost)
        initial_cost[name] = cost
    selected_actor = copy.deepcopy(actor).eval()
    selected_round = 0
    selected_selection_cost = float(initial_cost["selection"].mean())
    critic_rng = np.random.default_rng(833_000 + fold * 100 + seed)
    set_seed(833_000 + fold * 100 + seed)
    rounds = []
    args.defer_actor_projection = True
    for round_index in range(1, args.rounds + 1):
        actor_lr = actor_lr_for_round(args, round_index)
        for group in actor_optimizer.param_groups:
            group["lr"] = actor_lr
        exploration_decay_rounds = args.exploration_decay_rounds or args.rounds
        fraction = min(round_index - 1, exploration_decay_rounds - 1) / max(
            exploration_decay_rounds - 1, 1
        )
        radius = args.exploration_radius_start + fraction * (
            args.exploration_radius_end - args.exploration_radius_start
        )
        current_center = actor_predict(actor, inputs, fit, device)
        bank, cost, bank_diagnostics = make_two_stage_exploration_bank(
            data, current_center, fit, radius, args.second_radius_ratio,
            "search_recentered65", backend, weights, params, device,
            args.rollout_batch_size,
        )
        replay_actions = np.concatenate((replay_actions, bank), axis=1)
        replay_costs = np.concatenate((replay_costs, cost), axis=1)
        critic_logs = []
        for _ in range(args.critic_updates_per_round):
            for critic, optimizer, spec in zip(
                critics, critic_optimizers, training
            ):
                critic_logs.append(critic_update(
                    critic, optimizer, spec, inputs, fit, replay_actions,
                    replay_costs, critic_rng, args, device,
                ))
        before_parameters = {
            name: value.detach().clone() for name, value in actor.named_parameters()
        }
        before_action = actor_predict(actor, inputs, fit, device)
        actor_logs = []
        for microstep in range(k):
            actor_rng = np.random.default_rng(
                834_000_000 + fold * 100_000 + seed * 10_000
                + round_index * 100 + microstep
            )
            actor_logs.append(actor_update(
                actor, selected_actor, actor_optimizer, critics, training,
                inputs, fit, actor_rng, args, device,
            ))
        raw_step, projected_step, projection = project_round_parameters(
            actor, before_parameters, before_action, inputs, fit,
            args.max_round_step_sigma_rms, args.round_step_mode, device,
        )
        selection_metrics, _, selection_cost = evaluate_actor(
            actor, inputs, selection, data, initial_cost["selection"],
            backend, weights, params, device, args.rollout_batch_size,
        )
        if float(selection_cost.mean()) < selected_selection_cost:
            selected_selection_cost = float(selection_cost.mean())
            selected_round = round_index
            selected_actor = copy.deepcopy(actor).eval()
        oof_metrics = None
        if round_index in evaluation_rounds:
            oof_metrics, _, _ = evaluate_actor(
                actor, inputs, oof, data, initial_cost["oof"], backend, weights,
                params, device, args.rollout_batch_size,
            )
        rounds.append({
            "round": round_index, "radius_sigma": radius,
            "replay_candidates_per_state": int(replay_actions.shape[1]),
            "bank": bank_diagnostics,
            "critic_loss_mean": float(np.mean([row["loss"] for row in critic_logs])),
            "actor_microsteps": k,
            "actor_lr": actor_lr,
            "actor_raw_cumulative_step_sigma_rms": raw_step,
            "actor_projected_cumulative_step_sigma_rms": projected_step,
            "actor_cumulative_projection": projection,
            "actor_round_step_mode": args.round_step_mode,
            "actor_last_microstep": actor_logs[-1],
            "selection": selection_metrics,
            "oof": oof_metrics,
            "selected": bool(selected_round == round_index),
        })
        print(
            f"K={k} fold={fold} seed={seed} round={round_index} "
            f"raw={raw_step:.4f} projected={projected_step:.4f} "
            f"sel_delta={selection_metrics['gain_vs_pretrained_actor']['mean']:.1f} "
            f"selected={selected_round}", flush=True,
        )
    final = {"selected": {}, "latest": {}}
    saved_center = None
    saved_cost = None
    for role, model in (("selected", selected_actor), ("latest", actor)):
        for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
            metrics, center, cost = evaluate_actor(
                model, inputs, rows, data, initial_cost[name], backend, weights,
                params, device, args.rollout_batch_size,
            )
            final[role][name] = metrics
            if role == "selected" and name == "oof":
                saved_center, saved_cost = center, cost
    local_bank = make_exploration_bank(saved_center, 0.05)
    local_cost = rollout_bank(
        data, local_bank, oof, backend, weights, params, device,
        args.rollout_batch_size,
    )
    run_dir = output / f"k{k}" / f"fold{fold}_seed{seed}"
    run_dir.mkdir(parents=True)
    evaluation_path = run_dir / "evaluation.npz"
    np.savez_compressed(
        evaluation_path, fit_indices=fit, selection_indices=selection,
        oof_indices=oof, selected_oof_center=saved_center,
        selected_oof_cost=saved_cost, local_bank=local_bank, local_cost=local_cost,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({
        "qualification": "HIGHSPEED_ACTOR_K_SCAN_TRAIN_ONLY",
        "k": k, "fold": fold, "seed": seed, "selected_round": selected_round,
        "normalization": source["normalization"],
        "actor_selected_state_dict": selected_actor.state_dict(),
        "actor_latest_state_dict": actor.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic1_state_dict": critics[0].state_dict(),
        "critic2_state_dict": critics[1].state_dict(),
        "critic1_optimizer": critic_optimizers[0].state_dict(),
        "critic2_optimizer": critic_optimizers[1].state_dict(),
        "critic1_training": training[0], "critic2_training": training[1],
        "fit_indices": fit, "selection_indices": selection, "oof_indices": oof,
        "source_oac_checkpoint_sha256": source_record["checkpoint_sha256"],
        "source_budget_checkpoint_sha256": budget_record["checkpoint_sha256"],
        "formal_validation_or_test_created": False,
    }, checkpoint_path)
    return {
        "k": k, "selected_round": selected_round,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluation": str(evaluation_path.resolve()),
        "evaluation_sha256": sha256(evaluation_path),
        "initial": initial, "selected": final["selected"], "latest": final["latest"],
        "rounds": rounds,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.resolve()
    budget_root = args.budget_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_summary_path = source_root / "summary.json"
    source_validator_path = source_root / "validator_report.json"
    budget_summary_path = budget_root / "summary.json"
    budget_validator_path = budget_root / "validator_report.json"
    source_summary = json.loads(source_summary_path.read_text())
    budget_summary = json.loads(budget_summary_path.read_text())
    if json.loads(source_validator_path.read_text())["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("source validation missing")
    if json.loads(budget_validator_path.read_text())["qualification"] != "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_INDEPENDENT_RELOAD_PASS":
        raise AssertionError("budget validation missing")
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    budget_records = {
        (int(row["fold"]), int(row["seed"])): row for row in budget_summary["records"]
    }
    data = load_shared_data(json.loads((source_root / "contract.json").read_text()))
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    contract = {
        "format": "highspeed_actor_k_scan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_K_SCAN_CONTRACT_TRAIN_ONLY",
        "k_values": k_values, "folds": folds, "seeds": seeds,
        "initialization": "same R1 Actor/Replay plus absorbed1600 Twin Critic",
        "equal_budget": {
            "rollout_candidates_per_state_per_round": 65,
            "critic_updates_per_twin_per_round": args.critic_updates_per_round,
            "cumulative_actor_output_trust_sigma_rms": args.max_round_step_sigma_rms,
            "round_step_mode": args.round_step_mode,
            "only_variable": "Actor microsteps K",
        },
        "analytic_dbm_gradient": False,
        "formal_validation_or_test_created": False,
        "source_summary_sha256": sha256(source_summary_path),
        "source_validator_sha256": sha256(source_validator_path),
        "budget_summary_sha256": sha256(budget_summary_path),
        "budget_validator_sha256": sha256(budget_validator_path),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    records = []
    for fold in folds:
        for seed in seeds:
            key = (fold, seed)
            arms = {}
            for k in k_values:
                arms[str(k)] = run_arm(
                    k, fold, seed, args, data, source_records[key],
                    budget_records[key], output, device,
                )
            records.append({"fold": fold, "seed": seed, "arms": arms})
    summary = {
        "format": "highspeed_actor_k_scan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_K_SCAN_COMPLETE_TRAIN_ONLY",
        "contract_sha256": sha256(output / "contract.json"),
        "aggregate": {str(k): aggregate(records, k) for k in k_values},
        "records": records,
        "formal_validation_or_test_created": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
