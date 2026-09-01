#!/usr/bin/env python3
"""Freeze Actor and extend Twin-Critic training on search-informed Replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import load_data
from train_highspeed_actor_visited_oac import (
    build_inputs,
    critic_update,
    local_critic_metrics,
)


DEFAULT_SOURCE = Path(
    "outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1"
)
DEFAULT_BANKS = Path(
    "outputs/mppi_proposal/highspeed_search_replay_critic_ab_20260830_v1/treatment_banks.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fresh-banks", type=Path, default=DEFAULT_BANKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="0,400,1600")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--critic-state-batch-size", type=int, default=8)
    parser.add_argument("--critic-candidates-per-state", type=int, default=48)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size), "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


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


def evaluate(
    critics: tuple[ConfigurableAbsoluteActionValueCritic, ...],
    training: tuple[dict, ...], inputs: tuple[np.ndarray, ...], oof: np.ndarray,
    path_bank: np.ndarray, path_cost: np.ndarray,
    local_bank: np.ndarray, local_cost: np.ndarray, device: torch.device,
) -> dict[str, Any]:
    return {
        "path": local_critic_metrics(
            critics, training, inputs, oof, path_bank, path_cost, device
        ),
        "local_0p05": local_critic_metrics(
            critics, training, inputs, oof, local_bank, local_cost, device
        ),
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.resolve()
    banks_path = args.fresh_banks.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or budgets[0] != 0:
        raise AssertionError("budgets must include zero")
    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    seeds = {int(value) for value in args.seeds.split(",") if value.strip()}
    contract_path = source_root / "contract.json"
    summary_path = source_root / "summary.json"
    validator_path = source_root / "validator_report.json"
    contract = json.loads(contract_path.read_text())
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    if validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("source OAC independent replay did not pass")
    if contract["exploration"]["mode"] != "search_recentered65":
        raise AssertionError("source is not search-informed Replay")
    if contract["analytic_dbm_gradient"]:
        raise AssertionError("source used analytic DBM gradient")
    data = load_source_data(contract)
    with np.load(banks_path, allow_pickle=False) as loaded:
        bank_data = {name: np.asarray(loaded[name]) for name in loaded.files}
    bank_map = {
        (int(fold), int(seed)): index
        for index, (fold, seed) in enumerate(zip(bank_data["fold"], bank_data["seed"]))
    }
    records_by_key = {
        (int(row["fold"]), int(row["seed"])): row for row in summary["records"]
    }
    selected_keys = [key for key in sorted(records_by_key) if key[0] in folds and key[1] in seeds]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output.mkdir(parents=True)
    records = []
    for fold, seed in selected_keys:
        expected = records_by_key[(fold, seed)]
        checkpoint = Path(expected["checkpoint"])
        replay_path = Path(expected["replay"])
        if sha256(checkpoint) != expected["checkpoint_sha256"]:
            raise AssertionError("source checkpoint hash mismatch")
        if sha256(replay_path) != expected["replay_sha256"]:
            raise AssertionError("source replay hash mismatch")
        payload = torch.load(checkpoint, map_location=device)
        fit = np.asarray(payload["fit_indices"], np.int64)
        oof = np.asarray(payload["oof_indices"], np.int64)
        inputs = build_inputs(data, payload)
        critics = []
        optimizers = []
        training = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            optimizer = torch.optim.AdamW(
                critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay
            )
            optimizer.load_state_dict(payload[f"critic{twin}_optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.critic_lr
                group["weight_decay"] = args.weight_decay
            critics.append(critic); optimizers.append(optimizer)
            training.append(payload[f"critic{twin}_training"])
        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {name: np.asarray(loaded[name]) for name in loaded.files}
        if not np.array_equal(replay["fit_indices"], fit):
            raise AssertionError("fit index mismatch")
        bank_index = bank_map[(fold, seed)]
        if not np.array_equal(bank_data["oof"][bank_index], oof):
            raise AssertionError("fresh bank OOF index mismatch")
        path_bank = bank_data["bank"][bank_index]
        path_cost = bank_data["cost"][bank_index]
        local_bank = replay["oof_probe_bank"]
        local_cost = replay["oof_probe_cost"]
        random.seed(831_000 + fold * 100 + seed)
        np.random.seed(831_000 + fold * 100 + seed)
        torch.manual_seed(831_000 + fold * 100 + seed)
        rng = np.random.default_rng(831_000 + fold * 100 + seed)
        curve = []
        previous = 0
        for budget in budgets:
            logs = []
            for _ in range(budget - previous):
                for critic, optimizer, spec in zip(critics, optimizers, training):
                    logs.append(critic_update(
                        critic, optimizer, spec, inputs, fit, replay["actions"],
                        replay["costs"], rng, args, device,
                    ))
            metric = evaluate(
                tuple(critics), tuple(training), inputs, oof, path_bank,
                path_cost, local_bank, local_cost, device,
            )
            curve.append({
                "extra_updates_per_twin": budget,
                "train": None if not logs else {
                    "loss_mean": float(np.mean([row["loss"] for row in logs])),
                    "value_mean": float(np.mean([row["value"] for row in logs])),
                    "ranking_mean": float(np.mean([row["ranking"] for row in logs])),
                },
                **metric,
            })
            print(
                f"fold={fold} seed={seed} budget={budget} "
                f"path_sign={metric['path']['center_relative_sign_accuracy']:.4f} "
                f"path_recovery={metric['path']['bank_gain_recovery']:.4f}",
                flush=True,
            )
            previous = budget
        checkpoint_out = output / f"critic_fold{fold}_seed{seed}.pt"
        torch.save({
            "qualification": "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_TRAIN_ONLY",
            "fold": fold, "seed": seed, "extra_updates_per_twin": budgets[-1],
            "critic1_state_dict": critics[0].state_dict(),
            "critic2_state_dict": critics[1].state_dict(),
            "critic1_optimizer": optimizers[0].state_dict(),
            "critic2_optimizer": optimizers[1].state_dict(),
            "critic1_training": training[0], "critic2_training": training[1],
            "fit_indices": fit, "oof_indices": oof,
            "source_checkpoint_sha256": sha256(checkpoint),
            "formal_validation_or_test_created": False,
        }, checkpoint_out)
        records.append({
            "fold": fold, "seed": seed, "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": sha256(checkpoint),
            "checkpoint": str(checkpoint_out.resolve()),
            "checkpoint_sha256": sha256(checkpoint_out), "curve": curve,
        })
    metric_names = (
        "centered_log_cost_pearson", "center_relative_sign_accuracy",
        "bank_gain_recovery", "selected_beats_center_fraction",
    )
    aggregate = {}
    for budget in budgets:
        aggregate[str(budget)] = {}
        for bank_name in ("path", "local_0p05"):
            aggregate[str(budget)][bank_name] = {}
            for name in metric_names:
                values = [
                    next(row for row in record["curve"] if row["extra_updates_per_twin"] == budget)[bank_name][name]
                    for record in records
                ]
                aggregate[str(budget)][bank_name][name] = distribution(values)
    result = {
        "format": "highspeed_search_replay_critic_budget_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_COMPLETE_TRAIN_ONLY",
        "contract": {
            "actor_frozen": True, "extra_updates_per_twin": budgets,
            "new_rollout_count": 0, "analytic_dbm_gradient": False,
            "formal_validation_or_test_created": False,
            "folds": sorted(folds), "seeds": sorted(seeds),
        },
        "sources": {
            "oac_summary": str(summary_path), "oac_summary_sha256": sha256(summary_path),
            "oac_validator": str(validator_path), "oac_validator_sha256": sha256(validator_path),
            "fresh_banks": str(banks_path), "fresh_banks_sha256": sha256(banks_path),
        },
        "aggregate": aggregate, "records": records,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualification": result["qualification"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
