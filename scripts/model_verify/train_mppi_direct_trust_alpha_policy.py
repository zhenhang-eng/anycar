#!/usr/bin/env python3
"""Train TR2-B move/stay plus continuous-alpha policy from verified TR1 lines.

The old/proposal trust direction is frozen.  Parameters are updated only from
``internal_fit`` episodes and epoch/seed are selected only on
``internal_selection`` unique-output DBM cost.  Formal validation and test are not
loaded.  The full 21-point line costs remain ready for the next Alpha Actor--Critic
stage but are not used as Critic targets here.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import TorchMPPITrustAlphaPolicy
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_trust_region_actor import (
    DEFAULT_LABELS,
    DEFAULT_OLD_ACTOR,
    TrustDataset,
    direct_cost,
    distribution,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_OUTPUT = Path("outputs/mppi_proposal/direct_trust_alpha_policy_20260810_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--old-actor", type=Path, default=DEFAULT_OLD_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--stay-class-weight", type=float, default=4.0)
    parser.add_argument("--alpha-loss-weight", type=float, default=1.0)
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--move-threshold", type=float, default=0.5)
    parser.add_argument("--improvement-cap", type=float, default=20.0)
    parser.add_argument("--regression-mean-penalty", type=float, default=0.25)
    parser.add_argument("--regression-p95-penalty", type=float, default=0.05)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_policy(old_payload: dict[str, Any], device: torch.device, dropout: float) -> TorchMPPITrustAlphaPolicy:
    policy = TorchMPPITrustAlphaPolicy(dropout=dropout).to(device)
    policy.load_actor_encoder_state_dict(old_payload["actor_state_dict"])
    return policy


def extra_tensors(data: TrustDataset, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "old_center": torch.from_numpy(data.old_center).to(device),
        "safe_center": torch.from_numpy(data.safe_center).to(device),
        "endpoint_center": torch.from_numpy(data.endpoint_center).to(device),
        "direction": torch.from_numpy(data.projected_direction).to(device),
        "rho": torch.from_numpy(data.requested_rho[:, None]).to(device),
        "scale": torch.from_numpy(data.trust_scale[:, None]).to(device),
        "sigma": torch.from_numpy(data.sigma).to(device),
        "safe_alpha": torch.from_numpy(data.safe_alpha).to(device),
        "move": torch.from_numpy((data.safe_index > 0).astype(np.float32)).to(device),
    }


def policy_batch(policy, tensors, extra, index: torch.Tensor):
    return policy(
        *(value[index] for value in tensors["inputs"]),
        extra["direction"][index], extra["rho"][index], extra["scale"][index],
    )


def center_from_alpha(extra: dict[str, torch.Tensor], index: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    raw = (
        extra["old_center"][index]
        + alpha[:, None, None]
        * extra["sigma"][index, None, :]
        * extra["direction"][index]
    )
    return torch.clamp(raw, -1.0, 1.0)


@torch.no_grad()
def outputs(policy, tensors, extra, index: np.ndarray, args, device):
    probabilities, conditional, hard, centers = [], [], [], []
    policy.eval()
    for start in range(0, len(index), args.evaluation_batch_size):
        one = torch.from_numpy(index[start:start + args.evaluation_batch_size]).to(device)
        _, probability, alpha = policy_batch(policy, tensors, extra, one)
        hard_alpha = policy.hard_alpha(probability, alpha, args.move_threshold)
        center = center_from_alpha(extra, one, hard_alpha)
        probabilities.append(probability.cpu().numpy())
        conditional.append(alpha.cpu().numpy())
        hard.append(hard_alpha.cpu().numpy())
        centers.append(center.cpu().numpy())
    return tuple(np.concatenate(value) for value in (probabilities, conditional, hard, centers))


def evaluate(policy, data, tensors, extra, index: np.ndarray, args, device) -> dict[str, Any]:
    probability, conditional, alpha, center = outputs(
        policy, tensors, extra, index, args, device
    )
    cost = direct_cost(center, data, tensors, index, args.evaluation_batch_size, device)
    old = data.old_cost[index]
    gain = old - cost
    regression = np.maximum(-gain, 0.0)
    target_move = data.safe_index[index] > 0
    predicted_move = probability > args.move_threshold
    move_recall = float(np.mean(predicted_move[target_move])) if np.any(target_move) else 1.0
    stay_recall = float(np.mean(~predicted_move[~target_move])) if np.any(~target_move) else 1.0
    move_precision = (
        float(np.mean(target_move[predicted_move])) if np.any(predicted_move) else 1.0
    )
    move_mask = target_move
    alpha_mae = float(np.mean(np.abs(conditional[move_mask] - data.safe_alpha[index][move_mask])))
    center_error = (center - data.safe_center[index]) / data.sigma[index, None, :]
    score = (
        float(np.mean(cost))
        + args.regression_mean_penalty * float(np.mean(regression))
        + args.regression_p95_penalty * float(np.quantile(regression, 0.95))
    )
    return {
        "selection_score": score,
        "direct_cost": distribution(cost),
        "old_cost": distribution(old),
        "safe_cost": distribution(data.safe_cost[index]),
        "gain_vs_old": distribution(gain),
        "regression_fraction": float(np.mean(gain < 0.0)),
        "move_fraction": float(np.mean(predicted_move)),
        "move_accuracy": float(np.mean(predicted_move == target_move)),
        "move_recall": move_recall,
        "stay_recall": stay_recall,
        "move_precision": move_precision,
        "conditional_alpha_mae_on_move": alpha_mae,
        "hard_alpha_mean": float(np.mean(alpha)),
        "hard_alpha_p95": float(np.quantile(alpha, 0.95)),
        "target_sigma_rmse": float(np.sqrt(np.mean(center_error ** 2))),
    }


def training_weights(data: TrustDataset, index: np.ndarray, args) -> np.ndarray:
    gain = np.maximum(data.old_cost[index] - data.safe_cost[index], 0.0)
    improvement = 1.0 + np.minimum(gain / max(args.improvement_cap, 1e-6), 1.0)
    classes = np.where(data.safe_index[index] == 0, args.stay_class_weight, 1.0)
    value = data.episode_weight[index] * improvement * classes
    return (value / np.mean(value)).astype(np.float32)


def train_seed(seed, old_payload, data, tensors, extra, fit_index, selection_index, args, device):
    set_seed(seed)
    policy = make_policy(old_payload, device, dropout=0.05)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = torch.from_numpy(training_weights(data, fit_index, args)).to(device)
    rng = np.random.default_rng(seed)
    history = []
    initial = evaluate(policy, data, tensors, extra, selection_index, args, device)
    initial.update({"epoch": 0, "learning_rate": args.learning_rate})
    history.append(initial)
    best_score = float(initial["selection_score"])
    best_epoch = 0
    best_state = copy.deepcopy(policy.state_dict())
    print(f"[seed={seed} epoch=000] score={best_score:.4f} cost={initial['direct_cost']['mean']:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        policy.train()
        order = rng.permutation(len(fit_index))
        losses = []
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(fit_index[local_np]).to(device)
            move_logit, probability, conditional = policy_batch(policy, tensors, extra, absolute)
            target_move = extra["move"][absolute]
            target_alpha = extra["safe_alpha"][absolute]
            bce = F.binary_cross_entropy_with_logits(move_logit, target_move, reduction="none")
            moving = target_move > 0.5
            alpha_loss = torch.zeros_like(bce)
            if torch.any(moving):
                alpha_loss[moving] = F.smooth_l1_loss(
                    conditional[moving], target_alpha[moving], beta=args.huber_beta, reduction="none"
                )
            soft_alpha = probability * conditional
            soft_center = center_from_alpha(extra, absolute, soft_alpha)
            center_error = (soft_center - extra["safe_center"][absolute]) / extra["sigma"][absolute, None, :]
            center_loss = F.smooth_l1_loss(
                center_error, torch.zeros_like(center_error), beta=args.huber_beta, reduction="none"
            ).mean(dim=(1, 2))
            per_sample = bce + args.alpha_loss_weight * alpha_loss + args.center_loss_weight * center_loss
            loss = torch.sum(per_sample * weights[local]) / torch.sum(weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.evaluation_interval == 0 or epoch == args.epochs:
            metrics = evaluate(policy, data, tensors, extra, selection_index, args, device)
            metrics.update({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            history.append(metrics)
            score = float(metrics["selection_score"])
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(policy.state_dict())
            gain = metrics["gain_vs_old"]
            print(
                f"[seed={seed} epoch={epoch:03d}] score={score:.4f} cost={metrics['direct_cost']['mean']:.4f} "
                f"p05={gain['p05']:.3f} worst={gain['minimum']:.3f} "
                f"stay={metrics['stay_recall']:.3f} move={metrics['move_recall']:.3f}", flush=True,
            )
    return best_state, best_epoch, history


def refit(seed, epoch_count, old_payload, data, tensors, extra, index, args, device):
    set_seed(seed)
    policy = make_policy(old_payload, device, dropout=0.05)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = torch.from_numpy(training_weights(data, index, args)).to(device)
    rng = np.random.default_rng(seed)
    for _ in range(epoch_count):
        policy.train()
        order = rng.permutation(len(index))
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(index[local_np]).to(device)
            move_logit, probability, conditional = policy_batch(policy, tensors, extra, absolute)
            target_move = extra["move"][absolute]
            target_alpha = extra["safe_alpha"][absolute]
            bce = F.binary_cross_entropy_with_logits(move_logit, target_move, reduction="none")
            moving = target_move > 0.5
            alpha_loss = torch.zeros_like(bce)
            alpha_loss[moving] = F.smooth_l1_loss(
                conditional[moving], target_alpha[moving], beta=args.huber_beta, reduction="none"
            )
            soft_center = center_from_alpha(extra, absolute, probability * conditional)
            error = (soft_center - extra["safe_center"][absolute]) / extra["sigma"][absolute, None, :]
            center_loss = F.smooth_l1_loss(
                error, torch.zeros_like(error), beta=args.huber_beta, reduction="none"
            ).mean(dim=(1, 2))
            per = bce + args.alpha_loss_weight * alpha_loss + args.center_loss_weight * center_loss
            loss = torch.sum(per * weights[local]) / torch.sum(weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
    policy.eval()
    return policy


def payload(policy, old_payload, args, seed, epoch, fit_episodes, selection_episodes, qualification):
    return {
        "format_version": 1,
        "method": "TR2-B deterministic move/stay and continuous trust alpha",
        "qualification": qualification,
        "policy_class": "TorchMPPITrustAlphaPolicy",
        "policy_state_dict": copy.deepcopy(policy.cpu().state_dict()),
        "state_normalization": old_payload["state_normalization"],
        "feedback_mean": old_payload["feedback_mean"],
        "feedback_std": old_payload["feedback_std"],
        "gradient_mean": old_payload["gradient_mean"],
        "gradient_std": old_payload["gradient_std"],
        "old_actor": str(args.old_actor.resolve()),
        "old_actor_sha256": sha256_file(args.old_actor),
        "proposal_actor": json.loads((args.labels / "config.json").read_text())["proposal_actor"],
        "proposal_actor_sha256": json.loads((args.labels / "config.json").read_text())["proposal_actor_sha256"],
        "labels": str(args.labels.resolve()),
        "labels_hashes": {name: sha256_file(args.labels / name) for name in ("config.json", "splits.json", "summary.json")},
        "move_threshold": args.move_threshold,
        "seed": seed,
        "selected_epoch": epoch,
        "fit_episodes": fit_episodes,
        "selection_episodes": selection_episodes,
        "training_arguments": vars(args),
        "test_policy": "formal validation and test not loaded or evaluated",
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    old_payload = load_actor_payload(args.old_actor)
    data, label_config, splits = load_dataset(args.labels, old_payload, args.max_snapshots)
    if label_config["old_actor_sha256"] != sha256_file(args.old_actor):
        raise AssertionError("old Actor differs from TR1 source")
    if args.max_snapshots:
        present = sorted(set(data.episodes.tolist()))
        fit_episodes = selection_episodes = present
    else:
        fit_episodes = list(splits["internal_fit"])
        selection_episodes = list(splits["internal_selection"])
    fit_index = np.flatnonzero(np.isin(data.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if not args.max_snapshots and len(fit_index) + len(selection_index) != len(data.episodes):
        raise AssertionError("internal split does not cover TR1 contexts")
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    runs = []
    for seed in args.seeds:
        state, epoch, history = train_seed(
            seed, old_payload, data, tensors, extra, fit_index, selection_index, args, device
        )
        policy = make_policy(old_payload, device, dropout=0.0)
        policy.load_state_dict(state, strict=True)
        metrics = evaluate(policy, data, tensors, extra, selection_index, args, device)
        torch.save(payload(
            policy, old_payload, args, seed, epoch, fit_episodes, selection_episodes,
            "TR2B_INTERNAL_SELECTION_ONLY",
        ), args.output_dir / f"selection_alpha_seed{seed}.pt")
        runs.append({"seed": seed, "selected_epoch": epoch, "selected_metrics": metrics, "history": history})
    winner = min(runs, key=lambda row: row["selected_metrics"]["selection_score"])
    selected_seed, selected_epoch = int(winner["seed"]), int(winner["selected_epoch"])
    all_index = np.arange(len(data.episodes), dtype=np.int64)
    final = refit(selected_seed, selected_epoch, old_payload, data, tensors, extra, all_index, args, device)
    final_metrics = evaluate(final, data, tensors, extra, all_index, args, device)
    final_path = args.output_dir / "trust_alpha_policy_selected.pt"
    torch.save(payload(
        final, old_payload, args, selected_seed, selected_epoch, fit_episodes, selection_episodes,
        "TR2B_FROZEN_TRAIN_DOMAIN_ONLY",
    ), final_path)
    summary = {
        "format_version": 1,
        "method": "TR2-B deterministic move/stay and continuous trust alpha",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "snapshot_count": len(data.episodes) // 2,
        "context_count": len(data.episodes),
        "fit_episode_count": len(fit_episodes),
        "selection_episode_count": len(selection_episodes),
        "fit_context_count": len(fit_index),
        "selection_context_count": len(selection_index),
        "runs": runs,
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "selected_internal_metrics": winner["selected_metrics"],
        "refit_all_train_episodes": True,
        "final_all_train_metrics": final_metrics,
        "policy_parameter_count": final.parameter_count,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": sha256_file(final_path),
        "qualification": "TR2B_FROZEN_TRAIN_DOMAIN_ONLY",
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "selected_internal_metrics": winner["selected_metrics"],
        "final_all_train_metrics": final_metrics,
        "policy_parameter_count": final.parameter_count,
        "checkpoint": str(final_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
