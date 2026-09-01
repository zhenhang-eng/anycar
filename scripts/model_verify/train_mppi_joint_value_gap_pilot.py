#!/usr/bin/env python3
"""Jointly continue Twin value Critics and a continuous relative-gap head.

This is an Actor-frozen, train-only mechanism pilot.  It reuses the fixed OAC-1
Replay and the 6400-update Critics, performs no DBM rollout, and never loads the
formal validation/test splits.  The auxiliary head predicts log1p of the raw
cost gap to a reference candidate; its loss backpropagates into both Critics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from analyze_mppi_online_absolute_sac_oac1 import calibrated_flat_gate
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
    metrics as bank_metrics,
)
from run_mppi_oac1_fixed_replay_curve import (
    load_value_checkpoint,
    predict_bank,
)
from train_mppi_online_absolute_sac import (
    critic_state_inputs,
    load_bank,
    material_pair_accuracy,
    predict_actions,
    sample_pairs,
    sample_training_points,
)


DEFAULT_PARENT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_extended_20260820_v1"
)
DEFAULT_BANK = Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_20260820_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--checkpoints", default="0,400,800,1600,3200")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--gap-batch-size", type=int, default=128)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gap-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--gap-weight", type=float, default=0.05)
    parser.add_argument(
        "--target-mode",
        choices=("move_coefficient", "signed_delta", "nonnegative_gap"),
        default="move_coefficient",
    )
    parser.add_argument("--material-gap", type=float, default=0.1)
    parser.add_argument("--flat-gap", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


class ContinuousGapHead(nn.Module):
    """Calibrate Twin log-values and an action pair into log1p(raw cost gap)."""

    def __init__(self, output_mode: str) -> None:
        super().__init__()
        self.output_mode = output_mode
        # q1/q2 for candidate/reference, deltas, Twin disagreement, 16-D action delta.
        self.network = nn.Sequential(
            nn.Linear(24, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, q1: torch.Tensor, q2: torch.Tensor,
        candidate: torch.Tensor, reference: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((
            q1, q2,
            q1[:, :1] - q1[:, 1:], q2[:, :1] - q2[:, 1:],
            (q1[:, :1] - q2[:, :1]).abs(),
            (q1[:, 1:] - q2[:, 1:]).abs(),
            (candidate - reference).reshape(len(candidate), 16),
        ), dim=1)
        output = self.network(features)[:, 0]
        if self.output_mode == "signed_delta":
            return output
        if self.output_mode == "move_coefficient":
            return torch.sigmoid(output)
        return F.softplus(output)


def physical_log_value(
    model: AbsoluteActionValueCritic, payload: dict[str, Any],
    inputs: tuple[np.ndarray, ...], state: np.ndarray, actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    prediction = model(
        torch.from_numpy(inputs[0][state]).to(device),
        torch.from_numpy(inputs[1][state]).to(device),
        torch.from_numpy(inputs[2][state]).to(device),
        torch.from_numpy(actions).to(device),
    )
    return (
        prediction * float(payload["training"]["target_std"])
        + float(payload["training"]["target_mean"])
    )


def value_objective(
    model: AbsoluteActionValueCritic, payload: dict[str, Any],
    inputs: tuple[np.ndarray, ...], points, pairs, args, device,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, action, cost = points
    pair_state, pair_action, pair_cost = pairs
    mean = float(payload["training"]["target_mean"])
    std = float(payload["training"]["target_std"])
    prediction = model(
        torch.from_numpy(inputs[0][state]).to(device),
        torch.from_numpy(inputs[1][state]).to(device),
        torch.from_numpy(inputs[2][state]).to(device),
        torch.from_numpy(action[:, None]).to(device),
    )[:, 0]
    target = (torch.log1p(torch.from_numpy(cost).to(device)) - mean) / std
    value = F.smooth_l1_loss(prediction, target)
    pair_prediction = model(
        torch.from_numpy(inputs[0][pair_state]).to(device),
        torch.from_numpy(inputs[1][pair_state]).to(device),
        torch.from_numpy(inputs[2][pair_state]).to(device),
        torch.from_numpy(pair_action).to(device),
    )
    raw_delta = torch.from_numpy(pair_cost[:, 0] - pair_cost[:, 1]).to(device)
    predicted_delta = pair_prediction[:, 0] - pair_prediction[:, 1]
    ranking = F.softplus(
        -raw_delta.sign() * predicted_delta / args.ranking_temperature
    ).mean()
    loss = value + args.ranking_weight * ranking
    return loss, {"value": float(value.detach()), "ranking": float(ranking.detach())}


def group_best_rows(replay: dict[str, np.ndarray]) -> np.ndarray:
    result = np.empty(len(replay["cost"]), np.int64)
    for group in np.unique(replay["interaction_group"]):
        members = np.flatnonzero(replay["interaction_group"] == group)
        best = members[np.argmin(replay["cost"][members])]
        result[members] = best
    return result


def sample_gap_batch(
    data: dict[str, np.ndarray], train: np.ndarray,
    replay: dict[str, np.ndarray], replay_best: np.ndarray,
    count: int, rng: np.random.Generator, target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    historical = count // 2
    online = count - historical
    state_h = rng.choice(train, historical, replace=True)
    candidate_h = rng.integers(24, size=historical)
    true_best_h = np.argmin(data["costs"][state_h], axis=1)
    best_h = (
        rng.integers(24, size=historical)
        if target_mode in ("signed_delta", "move_coefficient")
        else true_best_h
    )
    if target_mode == "move_coefficient":
        roles = rng.integers(3, size=historical)
        candidate_h[roles == 0] = true_best_h[roles == 0]
        best_h[roles == 1] = true_best_h[roles == 1]
    action_h = data["actions"][state_h, candidate_h]
    reference_h = data["actions"][state_h, best_h]
    gap_h = data["costs"][state_h, candidate_h] - data["costs"][state_h, best_h]
    row = rng.choice(len(replay["cost"]), online, replace=True)
    if target_mode in ("signed_delta", "move_coefficient"):
        best_row = np.empty_like(row)
        for position, source in enumerate(row):
            members = np.flatnonzero(
                replay["interaction_group"] == replay["interaction_group"][source]
            )
            best_row[position] = rng.choice(members)
        if target_mode == "move_coefficient":
            roles = rng.integers(3, size=online)
            candidate_best = replay_best[row]
            row[roles == 0] = candidate_best[roles == 0]
            best_row[roles == 1] = candidate_best[roles == 1]
    else:
        best_row = replay_best[row]
    state_o = replay["state_index"][row]
    if not np.array_equal(state_o, replay["state_index"][best_row]):
        raise AssertionError("replay candidate/reference state mismatch")
    action_o = replay["action"][row]
    reference_o = replay["action"][best_row]
    gap_o = replay["cost"][row] - replay["cost"][best_row]
    if target_mode == "nonnegative_gap":
        gap_h = np.maximum(gap_h, 0.0)
        gap_o = np.maximum(gap_o, 0.0)
    return (
        np.concatenate((state_h, state_o)).astype(np.int64),
        np.concatenate((action_h, action_o)).astype(np.float32),
        np.concatenate((reference_h, reference_o)).astype(np.float32),
        np.concatenate((gap_h, gap_o)).astype(np.float32),
    )


def gap_objective(
    critic1, payload1, inputs1, critic2, payload2, inputs2, gap_head,
    batch, device, target_mode,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, candidate, reference, raw_gap = batch
    pair = np.stack((candidate, reference), axis=1)
    q1 = physical_log_value(critic1, payload1, inputs1, state, pair, device)
    q2 = physical_log_value(critic2, payload2, inputs2, state, pair, device)
    candidate_t = torch.from_numpy(candidate).to(device)
    reference_t = torch.from_numpy(reference).to(device)
    prediction = gap_head(q1, q2, candidate_t, reference_t)
    raw_target = torch.from_numpy(raw_gap).to(device)
    if target_mode == "signed_delta":
        target = raw_target.sign() * torch.log1p(raw_target.abs())
    elif target_mode == "move_coefficient":
        positive = raw_target.clamp_min(0.0)
        target = positive / (positive + 0.1)
    else:
        target = torch.log1p(raw_target)
    loss = F.smooth_l1_loss(prediction, target)
    return loss, {
        "gap": float(loss.detach()),
        "predicted_log_gap_mean": float(prediction.detach().mean()),
        "target_log_gap_mean": float(target.detach().mean()),
    }


def predict_gap_bank(
    data, states, critic1, payload1, inputs1, critic2, payload2, inputs2,
    gap_head, device, target_mode, batch_size=256,
) -> tuple[np.ndarray, np.ndarray]:
    q1 = predict_bank(critic1, inputs1, payload1, states, data["actions"][states], device)
    q2 = predict_bank(critic2, inputs2, payload2, states, data["actions"][states], device)
    conservative = np.maximum(q1, q2)
    reference_index = np.argmin(conservative, axis=1)
    result = []
    gap_head.eval()
    for begin in range(0, len(states) * 24, batch_size):
        flat = np.arange(begin, min(begin + batch_size, len(states) * 24))
        local_state = flat // 24
        candidate_index = flat % 24
        global_state = states[local_state]
        candidate = data["actions"][global_state, candidate_index]
        reference = data["actions"][global_state, reference_index[local_state]]
        pair = np.stack((candidate, reference), axis=1)
        with torch.no_grad():
            z1 = physical_log_value(
                critic1, payload1, inputs1, global_state, pair, device
            )
            z2 = physical_log_value(
                critic2, payload2, inputs2, global_state, pair, device
            )
            log_gap = gap_head(
                z1, z2, torch.from_numpy(candidate).to(device),
                torch.from_numpy(reference).to(device),
            )
            if target_mode == "signed_delta":
                physical = log_gap.sign() * torch.expm1(log_gap.abs())
            elif target_mode == "move_coefficient":
                coefficient = log_gap.clamp(0.0, 1.0 - 1e-5)
                physical = 0.1 * coefficient / (1.0 - coefficient)
            else:
                physical = torch.expm1(log_gap)
            result.append(physical.cpu().numpy())
    return np.concatenate(result).reshape(len(states), 24), conservative


def gap_metrics(
    prediction: np.ndarray, costs: np.ndarray,
) -> dict[str, Any]:
    truth = costs - costs.min(axis=1, keepdims=True)
    effective_prediction = np.maximum(prediction, 0.0)
    best = np.argmin(costs, axis=1)
    rows = np.arange(len(costs))
    warm_material = truth[:, 0] > 0.1
    best_prediction = prediction[rows, best]
    warm_prediction = prediction[:, 0]
    fixed_recall = float(np.mean(best_prediction <= 0.1))
    fixed_false = float(np.mean(warm_prediction[warm_material] <= 0.1))
    coefficient = effective_prediction / (effective_prediction + 0.1)
    coefficient_truth = truth / (truth + 0.1)
    return {
        "count": int(len(costs)),
        "log_gap_pearson": correlation(np.log1p(effective_prediction).ravel(), np.log1p(truth).ravel()),
        "log_gap_mae": float(np.mean(np.abs(np.log1p(effective_prediction) - np.log1p(truth)))),
        "move_coefficient_mae": float(np.mean(np.abs(coefficient - coefficient_truth))),
        "fixed_gap_0_1": {
            "bank_best_recall": fixed_recall,
            "warm_false_stay": fixed_false,
        },
        "bank_best_prediction": best_prediction,
        "warm_prediction": warm_prediction,
        "warm_material": warm_material,
    }


def calibrated_gap_gate(train_metrics: dict, heldout_metrics: dict) -> dict[str, Any]:
    train_best = train_metrics["bank_best_prediction"]
    train_warm = train_metrics["warm_prediction"][train_metrics["warm_material"]]
    thresholds = np.unique(np.concatenate((train_best, train_warm)))
    candidates = []
    for threshold in thresholds:
        recall = float(np.mean(train_best <= threshold))
        false_stay = float(np.mean(train_warm <= threshold))
        if false_stay <= 0.07 and recall >= 0.85:
            candidates.append((recall, -float(threshold), float(threshold), false_stay))
    if not candidates:
        return {"available": False, "rule": "train fs<=0.07 and recall>=0.85"}
    selected = max(candidates)
    threshold = selected[2]
    heldout_best = heldout_metrics["bank_best_prediction"]
    heldout_warm = heldout_metrics["warm_prediction"][heldout_metrics["warm_material"]]
    return {
        "available": True,
        "rule": "maximize train recall subject to fs<=0.07 and recall>=0.85",
        "threshold": threshold,
        "train": {"bank_best_recall": selected[0], "warm_false_stay": selected[3]},
        "heldout": {
            "bank_best_recall": float(np.mean(heldout_best <= threshold)),
            "warm_false_stay": float(np.mean(heldout_warm <= threshold)),
        },
    }


def strip_arrays(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)}


def bad_action_correction(
    replay: dict[str, np.ndarray], conservative: np.ndarray,
) -> dict[str, Any]:
    pre = np.maximum(replay["pre_critic1"], replay["pre_critic2"])
    group = replay["interaction_group"]
    eligible = replay["round"] <= int(replay["round"].max()) - 2
    bad_rows, mean_rows = [], []
    for key in np.unique(group[eligible]):
        members = np.flatnonzero(group == key)
        mean = members[replay["role"][members] == "actor_mean"]
        if len(mean) != 1:
            raise AssertionError("each interaction group must have one actor mean")
        bad = members[replay["cost"][members] > replay["cost"][mean[0]] + 0.1]
        bad_rows.extend(bad.tolist())
        mean_rows.extend([int(mean[0])] * len(bad))
    bad_rows = np.asarray(bad_rows, np.int64)
    mean_rows = np.asarray(mean_rows, np.int64)
    pre_correct = pre[bad_rows] > pre[mean_rows]
    final_correct = conservative[bad_rows] > conservative[mean_rows]
    initially_wrong = ~pre_correct
    return {
        "eligible_pairs": int(len(bad_rows)),
        "lag2_final_accuracy": float(np.mean(final_correct)),
        "initially_wrong_count": int(initially_wrong.sum()),
        "initially_wrong_corrected_fraction": float(
            np.mean(final_correct[initially_wrong]) if initially_wrong.any() else 1.0
        ),
    }


def evaluate(
    args, data, folds, replay, critic1, payload1, critic2, payload2,
    gap_head, device, outer_fold,
) -> dict[str, Any]:
    inputs1 = critic_state_inputs(data, payload1)
    inputs2 = critic_state_inputs(data, payload2)
    train = np.flatnonzero(folds != outer_fold)
    heldout = np.flatnonzero(folds == outer_fold)
    replay1 = predict_actions(
        critic1, inputs1, payload1, replay["state_index"], replay["action"], device
    )
    replay2 = predict_actions(
        critic2, inputs2, payload2, replay["state_index"], replay["action"], device
    )
    pair_accuracy, pair_count = material_pair_accuracy(
        np.maximum(replay1, replay2), replay["cost"], replay["interaction_group"],
        args.material_gap,
    )
    conservative_replay = np.maximum(replay1, replay2)
    by_speed = {}
    for speed in sorted(np.unique(data["speed"])):
        mask = data["speed"][replay["state_index"]] == speed
        accuracy, count = material_pair_accuracy(
            conservative_replay[mask], replay["cost"][mask],
            replay["interaction_group"][mask], args.material_gap,
        )
        by_speed[f"{speed:.1f}"] = {
            "material_pair_accuracy": accuracy,
            "material_pair_count": count,
        }
    train_gap, _ = predict_gap_bank(
        data, train, critic1, payload1, inputs1, critic2, payload2, inputs2,
        gap_head, device, args.target_mode,
    )
    heldout_gap, heldout_value = predict_gap_bank(
        data, heldout, critic1, payload1, inputs1, critic2, payload2, inputs2,
        gap_head, device, args.target_mode,
    )
    train_metrics = gap_metrics(train_gap, data["costs"][train])
    heldout_metrics = gap_metrics(heldout_gap, data["costs"][heldout])
    calibrated = calibrated_gap_gate(train_metrics, heldout_metrics)
    return {
        "actor_visited_material_pair_accuracy": pair_accuracy,
        "actor_visited_material_pair_count": pair_count,
        "actor_visited_by_speed": by_speed,
        "bad_action_correction": bad_action_correction(replay, conservative_replay),
        "heldout_bank": bank_metrics(heldout_value, data["costs"][heldout], args.material_gap),
        "gap_train": strip_arrays(train_metrics),
        "gap_heldout": strip_arrays(heldout_metrics),
        "gap_train_calibrated_heldout": calibrated,
    }


def save_checkpoint(path, update, critic1, critic2, gap_head, optimizer1, optimizer2, gap_optimizer):
    path.mkdir(parents=True, exist_ok=True)
    common = {
        "joint_additional_updates": update,
        "actor_update_count": 0,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    torch.save({**common, "model": critic1.state_dict(), "optimizer": optimizer1.state_dict()}, path / "critic1.pt")
    torch.save({**common, "model": critic2.state_dict(), "optimizer": optimizer2.state_dict()}, path / "critic2.pt")
    torch.save({**common, "model": gap_head.state_dict(), "optimizer": gap_optimizer.state_dict()}, path / "gap_head.pt")


def main() -> None:
    args = parse_args()
    checkpoints = [int(value) for value in args.checkpoints.split(",")]
    if checkpoints[0] != 0 or checkpoints != sorted(set(checkpoints)):
        raise AssertionError("checkpoints must be sorted unique and start at zero")
    seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device(args.device)
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    parent_contract = json.loads((args.parent_run / "contract.json").read_text())
    outer_fold = int(parent_contract.get("outer_fold", 0))
    if outer_fold not in (0, 1, 2):
        raise AssertionError("parent outer fold is not registered")
    source_oac1_run = Path(parent_contract["parent_run"])
    train = np.flatnonzero(folds != outer_fold)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "JOINT_VALUE_CONTINUOUS_GAP_PILOT_CONTRACT",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "parent_summary_sha256": sha256_file(args.parent_run / "summary.json"),
        "parent_contract_sha256": sha256_file(args.parent_run / "contract.json"),
        "source_oac1_run": str(source_oac1_run.resolve()),
        "outer_fold": outer_fold,
        "candidate_bank_sha256": sha256_file(args.bank_root / "candidate_bank.npz"),
        "gap_target": (
            "max(delta_J,0)/(max(delta_J,0)+0.1), balanced stay/move/random pairs"
            if args.target_mode == "move_coefficient"
            else (
                "sign(delta_J)*log1p(abs(delta_J)) for same-state action pairs"
                if args.target_mode == "signed_delta"
                else "log1p(max(J(candidate)-J(group_best),0))"
            )
        ),
        "gap_inputs": "Twin physical log-values, Twin disagreement, candidate-reference 16D delta",
        "joint_gradient": True,
        "new_dbm_rollouts": 0,
        "actor_module_constructed": False,
        "actor_optimizer_constructed": False,
        "actor_update_count": 0,
        "outer_fold_role": (
            "independent outer-split replication; no hyperparameter or threshold tuning"
            if outer_fold != 0
            else "consumed mechanism audit only; never checkpoint selection"
        ),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    records = []
    for seed in seeds:
        set_seed(26082100 + seed)
        rng = np.random.default_rng(26082100 + seed)
        parent = args.parent_run / f"seed_{seed}" / "updates_6400"
        critic1, payload1 = load_value_checkpoint(parent / "critic1.pt", device)
        critic2, payload2 = load_value_checkpoint(parent / "critic2.pt", device)
        inputs1 = critic_state_inputs(data, payload1)
        inputs2 = critic_state_inputs(data, payload2)
        if max(float(np.max(np.abs(a - b))) for a, b in zip(inputs1, inputs2)) > 1e-6:
            raise AssertionError("Twin Critic normalizations differ")
        replay_path = source_oac1_run / f"seed_{seed}" / "actor_visited_replay.npz"
        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        if not np.all(folds[replay["state_index"]] != outer_fold):
            raise AssertionError("Replay contains outer-heldout states")
        replay_best = group_best_rows(replay)
        gap_head = ContinuousGapHead(output_mode=args.target_mode).to(device)
        optimizer1 = torch.optim.AdamW(critic1.parameters(), lr=args.critic_learning_rate, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.AdamW(critic2.parameters(), lr=args.critic_learning_rate, weight_decay=args.weight_decay)
        # Restore the continuous 6400-step Value optimizer trajectory.
        optimizer1.load_state_dict(torch.load(parent / "critic1.pt", map_location=device)["optimizer"])
        optimizer2.load_state_dict(torch.load(parent / "critic2.pt", map_location=device)["optimizer"])
        gap_optimizer = torch.optim.AdamW(gap_head.parameters(), lr=args.gap_learning_rate, weight_decay=args.weight_decay)
        seed_dir = args.output_dir / f"seed_{seed}"
        points = []
        evaluation = evaluate(
            args, data, folds, replay, critic1, payload1, critic2, payload2,
            gap_head, device, outer_fold,
        )
        save_checkpoint(seed_dir / "updates_0", 0, critic1, critic2, gap_head, optimizer1, optimizer2, gap_optimizer)
        points.append({"additional_updates": 0, **evaluation})
        current = 0
        for target in checkpoints[1:]:
            loss_rows = []
            for _ in range(target - current):
                points_batch = sample_training_points(data, train, replay, 9, args.batch_size, rng)
                pairs = sample_pairs(data, train, replay, args.pair_batch_size, args.material_gap, rng)
                gap_batch = sample_gap_batch(
                    data, train, replay, replay_best, args.gap_batch_size, rng,
                    args.target_mode,
                )
                optimizer1.zero_grad(set_to_none=True)
                optimizer2.zero_grad(set_to_none=True)
                gap_optimizer.zero_grad(set_to_none=True)
                value1, info1 = value_objective(critic1, payload1, inputs1, points_batch, pairs, args, device)
                value2, info2 = value_objective(critic2, payload2, inputs2, points_batch, pairs, args, device)
                gap_loss, gap_info = gap_objective(
                    critic1, payload1, inputs1, critic2, payload2, inputs2,
                    gap_head, gap_batch, device, args.target_mode,
                )
                total = value1 + value2 + args.gap_weight * gap_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(critic1.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(critic2.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(gap_head.parameters(), 10.0)
                optimizer1.step(); optimizer2.step(); gap_optimizer.step()
                loss_rows.append({"total": float(total.detach()), **info1, **{f"critic2_{k}": v for k, v in info2.items()}, **gap_info})
            current = target
            evaluation = evaluate(
                args, data, folds, replay, critic1, payload1, critic2, payload2,
                gap_head, device, outer_fold,
            )
            save_checkpoint(seed_dir / f"updates_{target}", target, critic1, critic2, gap_head, optimizer1, optimizer2, gap_optimizer)
            points.append({
                "additional_updates": target,
                "mean_training_loss": {key: float(np.mean([row[key] for row in loss_rows])) for key in loss_rows[0]},
                **evaluation,
            })
            gate = evaluation["gap_train_calibrated_heldout"]
            held = gate.get("heldout", {})
            print(
                f"seed={seed} update={target} pair={evaluation['actor_visited_material_pair_accuracy']:.3f} "
                f"gapCorr={evaluation['gap_heldout']['log_gap_pearson']:.3f} "
                f"calR/FS={held.get('bank_best_recall', 0):.3f}/{held.get('warm_false_stay', 1):.3f}",
                flush=True,
            )
        replay_hash = sha256_file(replay_path)
        record = {
            "seed": seed,
            "source_replay": str(replay_path.resolve()),
            "source_replay_sha256": replay_hash,
            "points": points,
        }
        (seed_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    final = [record["points"][-1] for record in records]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "JOINT_VALUE_GAP_PENDING_ANALYSIS",
        "records": records,
        "final_actor_visited_pair_accuracy": [row["actor_visited_material_pair_accuracy"] for row in final],
        "final_gap_heldout_correlation": [row["gap_heldout"]["log_gap_pearson"] for row in final],
        "final_gap_heldout_fixed_gate": [row["gap_heldout"]["fixed_gap_0_1"] for row in final],
        "final_gap_calibrated_gate": [row["gap_train_calibrated_heldout"] for row in final],
        "actor_update_count": 0,
        "new_dbm_rollouts": 0,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    if args.target_mode == "move_coefficient":
        passes = [
            row["bank_best_recall"] >= 0.80
            and row["warm_false_stay"] <= 0.10
            for row in summary["final_gap_heldout_fixed_gate"]
        ]
    else:
        passes = [
            row.get("available", False)
            and row["heldout"]["bank_best_recall"] >= 0.80
            and row["heldout"]["warm_false_stay"] <= 0.10
            for row in summary["final_gap_calibrated_gate"]
        ]
    summary["joint_gate_pass_count"] = int(sum(passes))
    summary["qualification"] = (
        "JOINT_MOVE_COEFFICIENT_MECHANISM_PASS"
        if args.target_mode == "move_coefficient" and sum(passes) >= 2
        else (
            "JOINT_CONTINUOUS_GAP_MECHANISM_PASS"
            if sum(passes) >= 2 else "JOINT_CONTINUOUS_GAP_MECHANISM_FAIL"
        )
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "qualification", "final_actor_visited_pair_accuracy",
        "final_gap_heldout_correlation", "final_gap_heldout_fixed_gate",
        "final_gap_calibrated_gate", "joint_gate_pass_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
