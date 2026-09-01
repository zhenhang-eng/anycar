#!/usr/bin/env python3
"""Paired continuous-OAC transfer test for original versus absorbed Critics."""

from __future__ import annotations

import argparse
import copy
import json
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
from pretrain_highspeed_actor_twin_critic import actor_predict, distribution, load_data, sha256
from train_highspeed_actor_visited_oac import (
    actor_metrics,
    actor_update,
    build_inputs,
    critic_update,
    evaluate_actor,
    local_critic_metrics,
    make_exploration_bank,
    make_two_stage_exploration_bank,
    rollout_bank,
)


DEFAULT_SOURCE = Path("outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1")
DEFAULT_BUDGET = Path("outputs/mppi_proposal/highspeed_search_replay_critic_budget_20260830_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--budget-dir", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--critic-updates-per-round", type=int, default=20)
    parser.add_argument("--actor-updates-per-round", type=int, default=1)
    parser.add_argument("--critic-state-batch-size", type=int, default=8)
    parser.add_argument("--critic-candidates-per-state", type=int, default=48)
    parser.add_argument("--actor-batch-size", type=int, default=64)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.08)
    parser.add_argument("--exploration-radius-start", type=float, default=0.20)
    parser.add_argument("--exploration-radius-end", type=float, default=0.05)
    parser.add_argument("--second-radius-ratio", type=float, default=0.70)
    parser.add_argument("--max-round-step-sigma-rms", type=float, default=0.02)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--raw-cost-weight-cap", type=float, default=8.0)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shared_data(source_contract: dict) -> dict[str, np.ndarray]:
    root = Path(source_contract["source_pretrain"])
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


def tensors_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def make_critic(payload: dict, twin: int, device: torch.device) -> ConfigurableAbsoluteActionValueCritic:
    critic = ConfigurableAbsoluteActionValueCritic().to(device)
    critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
    return critic


def aggregate(records: list[dict[str, Any]], arm: str, split: str = "oof") -> dict[str, Any]:
    metrics = [row["arms"][arm]["selected"][split] for row in records]
    result = {}
    for name in (
        "teacher_gain_recovery", "beats_or_equals_warm_fraction",
        "regression_fraction_vs_warm", "beats_pretrained_fraction",
    ):
        result[name] = distribution([float(value[name]) for value in metrics])
    for gain_name in ("gain_vs_warm", "gain_vs_pretrained_actor"):
        result[gain_name] = {
            statistic: distribution([float(value[gain_name][statistic]) for value in metrics])
            for statistic in ("mean", "median", "p05", "min")
        }
    return result


def run_arm(
    arm: str, fold: int, seed: int, args: argparse.Namespace,
    data: dict[str, np.ndarray], source_record: dict, budget_record: dict,
    output: Path, device: torch.device,
) -> dict[str, Any]:
    source_checkpoint = Path(source_record["checkpoint"])
    source_replay_path = Path(source_record["replay"])
    if sha256(source_checkpoint) != source_record["checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    if sha256(source_replay_path) != source_record["replay_sha256"]:
        raise AssertionError("source replay hash mismatch")
    source = torch.load(source_checkpoint, map_location=device)
    if int(source["selected_round"]) != 20:
        raise AssertionError("paired transfer requires selected/latest round 20")
    if not tensors_equal(source["actor_selected_state_dict"], source["actor_latest_state_dict"]):
        raise AssertionError("selected and optimizer Actor states differ")
    with np.load(source_replay_path, allow_pickle=False) as loaded:
        source_replay = {name: np.asarray(loaded[name]) for name in loaded.files}
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

    critic_source = source
    critic_source_hash = source_record["checkpoint_sha256"]
    if arm == "absorbed1600":
        budget_checkpoint = Path(budget_record["checkpoint"])
        if sha256(budget_checkpoint) != budget_record["checkpoint_sha256"]:
            raise AssertionError("budget checkpoint hash mismatch")
        critic_source = torch.load(budget_checkpoint, map_location=device)
        critic_source_hash = budget_record["checkpoint_sha256"]
        if critic_source["source_checkpoint_sha256"] != source_record["checkpoint_sha256"]:
            raise AssertionError("budget/source lineage mismatch")
    critics = tuple(make_critic(critic_source, twin, device) for twin in (1, 2))
    training = tuple(critic_source[f"critic{twin}_training"] for twin in (1, 2))
    critic_optimizers = []
    for twin, critic in zip((1, 2), critics):
        optimizer = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(critic_source[f"critic{twin}_optimizer"])
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
    replay_actions = source_replay["actions"].copy()
    replay_costs = source_replay["costs"].copy()
    set_seed(832_000 + fold * 100 + seed)
    rng = np.random.default_rng(832_000 + fold * 100 + seed)
    rounds = []
    for round_index in range(1, args.rounds + 1):
        fraction = (round_index - 1) / max(args.rounds - 1, 1)
        radius = args.exploration_radius_start + fraction * (
            args.exploration_radius_end - args.exploration_radius_start
        )
        center = actor_predict(actor, inputs, fit, device)
        bank, cost, bank_diagnostics = make_two_stage_exploration_bank(
            data, center, fit, radius, args.second_radius_ratio,
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
                    replay_costs, rng, args, device,
                ))
        actor_logs = []
        for _ in range(args.actor_updates_per_round):
            actor_logs.append(actor_update(
                actor, selected_actor, actor_optimizer, critics, training,
                inputs, fit, rng, args, device,
            ))
        selection_metrics, _, selection_cost = evaluate_actor(
            actor, inputs, selection, data, initial_cost["selection"],
            backend, weights, params, device, args.rollout_batch_size,
        )
        if float(selection_cost.mean()) < selected_selection_cost:
            selected_selection_cost = float(selection_cost.mean())
            selected_round = round_index
            selected_actor = copy.deepcopy(actor).eval()
        rounds.append({
            "round": round_index, "radius_sigma": radius,
            "replay_candidates_per_state": int(replay_actions.shape[1]),
            "bank": bank_diagnostics,
            "critic_loss_mean": float(np.mean([value["loss"] for value in critic_logs])),
            "actor": actor_logs[-1], "selection": selection_metrics,
            "selected": bool(selected_round == round_index),
        })
        print(
            f"arm={arm} fold={fold} seed={seed} round={round_index:02d} "
            f"selection_delta={selection_metrics['gain_vs_pretrained_actor']['mean']:.1f} "
            f"selected={selected_round}", flush=True,
        )

    final = {"selected": {}, "latest": {}}
    centers = {"selected": {}, "latest": {}}
    costs = {"selected": {}, "latest": {}}
    for role, model in (("selected", selected_actor), ("latest", actor)):
        for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
            metrics, center, cost = evaluate_actor(
                model, inputs, rows, data, initial_cost[name], backend, weights,
                params, device, args.rollout_batch_size,
            )
            final[role][name] = metrics
            centers[role][name] = center
            costs[role][name] = cost
    path_bank, path_cost, _ = make_two_stage_exploration_bank(
        data, centers["selected"]["oof"], oof, 1.0,
        args.second_radius_ratio, "search_recentered65", backend, weights,
        params, device, args.rollout_batch_size,
    )
    local_bank = make_exploration_bank(centers["selected"]["oof"], 0.05)
    local_cost = rollout_bank(
        data, local_bank, oof, backend, weights, params, device,
        args.rollout_batch_size,
    )
    critic_metrics = {
        "path": local_critic_metrics(
            critics, training, inputs, oof, path_bank, path_cost, device
        ),
        "local_0p05": local_critic_metrics(
            critics, training, inputs, oof, local_bank, local_cost, device
        ),
    }
    run_dir = output / arm / f"fold{fold}_seed{seed}"
    run_dir.mkdir(parents=True)
    replay_path = run_dir / "replay.npz"
    np.savez_compressed(
        replay_path, fit_indices=fit, actions=replay_actions, costs=replay_costs,
        selected_oof_center=centers["selected"]["oof"],
        selected_oof_cost=costs["selected"]["oof"],
        path_bank=path_bank, path_cost=path_cost,
        local_bank=local_bank, local_cost=local_cost,
    )
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({
        "qualification": "HIGHSPEED_CRITIC_READINESS_ACTOR_TRANSFER_TRAIN_ONLY",
        "arm": arm, "fold": fold, "seed": seed,
        "selected_round": selected_round, "normalization": source["normalization"],
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
        "critic_initialization_sha256": critic_source_hash,
        "formal_validation_or_test_created": False,
    }, checkpoint)
    return {
        "selected_round": selected_round,
        "source_oac_checkpoint_sha256": source_record["checkpoint_sha256"],
        "critic_initialization_sha256": critic_source_hash,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "replay": str(replay_path.resolve()), "replay_sha256": sha256(replay_path),
        "initial": initial, "selected": final["selected"], "latest": final["latest"],
        "critic": critic_metrics, "rounds": rounds,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.resolve()
    budget_root = args.budget_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    source_summary_path = source_root / "summary.json"
    source_validator_path = source_root / "validator_report.json"
    budget_summary_path = budget_root / "summary.json"
    budget_validator_path = budget_root / "validator_report.json"
    source_summary = json.loads(source_summary_path.read_text())
    source_validator = json.loads(source_validator_path.read_text())
    budget_summary = json.loads(budget_summary_path.read_text())
    budget_validator = json.loads(budget_validator_path.read_text())
    if source_validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("source OAC validation missing")
    if budget_validator["qualification"] != "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_INDEPENDENT_RELOAD_PASS":
        raise AssertionError("Critic budget validation missing")
    source_contract = json.loads((source_root / "contract.json").read_text())
    if source_contract["exploration"]["mode"] != "search_recentered65":
        raise AssertionError("source is not search-informed OAC")
    data = load_shared_data(source_contract)
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    budget_records = {
        (int(row["fold"]), int(row["seed"])): row for row in budget_summary["records"]
    }
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    contract = {
        "format": "highspeed_critic_readiness_actor_transfer_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_CRITIC_READINESS_ACTOR_TRANSFER_CONTRACT_TRAIN_ONLY",
        "arms": ["original", "absorbed1600"],
        "only_initial_difference": "Twin Critic and optimizer before continuation",
        "continuation": {
            "rounds": args.rounds,
            "critic_actor_updates_per_round": [
                args.critic_updates_per_round, args.actor_updates_per_round
            ],
            "exploration": "search_recentered65",
            "radius_start_end_sigma": [
                args.exploration_radius_start, args.exploration_radius_end
            ],
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
            for arm in ("original", "absorbed1600"):
                arms[arm] = run_arm(
                    arm, fold, seed, args, data, source_records[key],
                    budget_records[key], output, device,
                )
            records.append({"fold": fold, "seed": seed, "arms": arms})
    paired = {}
    for metric_path, extractor in {
        "teacher_gain_recovery": lambda value: value["selected"]["oof"]["teacher_gain_recovery"],
        "warm_mean_gain": lambda value: value["selected"]["oof"]["gain_vs_warm"]["mean"],
        "warm_p05_gain": lambda value: value["selected"]["oof"]["gain_vs_warm"]["p05"],
        "beats_warm_fraction": lambda value: value["selected"]["oof"]["beats_or_equals_warm_fraction"],
        "path_sign": lambda value: value["critic"]["path"]["center_relative_sign_accuracy"],
        "path_recovery": lambda value: value["critic"]["path"]["bank_gain_recovery"],
    }.items():
        delta = [
            float(extractor(row["arms"]["absorbed1600"]))
            - float(extractor(row["arms"]["original"]))
            for row in records
        ]
        paired[metric_path] = distribution(delta)
    summary = {
        "format": "highspeed_critic_readiness_actor_transfer_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_CRITIC_READINESS_ACTOR_TRANSFER_COMPLETE_TRAIN_ONLY",
        "contract_sha256": sha256(output / "contract.json"),
        "record_count": len(records),
        "aggregate": {
            arm: aggregate(records, arm) for arm in ("original", "absorbed1600")
        },
        "paired_absorbed_minus_original": paired,
        "formal_validation_or_test_created": False,
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
