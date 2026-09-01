#!/usr/bin/env python3
"""Continue OAC-1 Critics on a frozen replay and measure a learning curve.

No DBM rollout is performed and no Actor module or optimizer is constructed.
The OAC-1 200-update checkpoint is evaluated as the baseline, then the two
value Critics and the unchanged flat/stay head continue to total update labels
400, 800, 1600, 3200 and 6400 using the original replay recipe and targets.

The parent run did not serialize optimizer state.  Therefore continuation uses
a freshly initialized AdamW optimizer at the registered OAC-1 learning rates;
this limitation is explicit in the contract and prevents interpreting the
curve as bitwise continuation of the original optimizer trajectory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
    metrics as bank_metrics,
)
from train_mppi_online_absolute_sac import (
    critic_state_inputs,
    final_metrics,
    load_bank,
    predict_actions,
    predict_flat,
    sample_flat_batch,
    sample_pairs,
    sample_training_points,
    update_flat_head,
    update_value_critic,
)
from analyze_mppi_online_absolute_sac_oac1 import calibrated_flat_gate


DEFAULT_PARENT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac01_20260820_v1"
)
DEFAULT_BANK = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_extended_20260820_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--checkpoints", default="200,400,800,1600,3200,6400")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flat-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
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


def serialized_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def load_value_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    if int(payload["actor_update_count"]) != 0:
        raise AssertionError(f"parent checkpoint contains Actor updates: {path}")
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def load_flat_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    if int(payload["actor_update_count"]) != 0:
        raise AssertionError(f"parent flat checkpoint contains Actor updates: {path}")
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def predict_bank(
    model, inputs, payload, states: np.ndarray, actions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    count, candidates = actions.shape[:2]
    return predict_actions(
        model, inputs, payload,
        np.repeat(states, candidates), actions.reshape(count * candidates, 8, 2),
        device,
    ).reshape(count, candidates)


def evaluate_checkpoint(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    replay: dict[str, np.ndarray],
    critic1: AbsoluteActionValueCritic,
    payload1: dict,
    critic2: AbsoluteActionValueCritic,
    payload2: dict,
    flat_head: AbsoluteActionValueCritic,
    flat_payload: dict,
    device: torch.device,
    outer_fold: int,
) -> dict:
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
    flat_probability = predict_flat(
        flat_head, flat_inputs, replay["state_index"], replay["action"], device
    )
    best_index = np.argmin(data["costs"], axis=1)
    all_states = np.arange(len(data["costs"]), dtype=np.int64)
    bank_best_action = data["actions"][all_states, best_index]
    flat_bank_best = predict_flat(
        flat_head, flat_inputs, all_states, bank_best_action, device
    )
    flat_warm = predict_flat(
        flat_head, flat_inputs, all_states, data["actions"][:, 0], device
    )
    core = final_metrics(
        data, replay, pred1, pred2, flat_probability,
        flat_bank_best, flat_warm, args.material_gap, args.flat_gap,
    )
    heldout = folds == outer_fold
    train = folds != outer_fold
    heldout_states = np.flatnonzero(heldout)
    heldout_actions = data["actions"][heldout]
    heldout_prediction = np.maximum(
        predict_bank(
            critic1, inputs1, payload1, heldout_states, heldout_actions, device
        ),
        predict_bank(
            critic2, inputs2, payload2, heldout_states, heldout_actions, device
        ),
    )
    heldout_bank = bank_metrics(
        heldout_prediction, data["costs"][heldout], args.material_gap
    )
    warm_material = (
        data["costs"][:, 0] - data["costs"].min(axis=1) > args.flat_gap
    )
    calibrated = calibrated_flat_gate(
        flat_bank_best, flat_warm, train, heldout, warm_material
    )
    conservative = np.maximum(pred1, pred2)
    by_speed = {}
    for speed in sorted(np.unique(data["speed"])):
        mask = data["speed"][replay["state_index"]] == speed
        # Preserve original interaction ids after subsetting.
        from train_mppi_online_absolute_sac import material_pair_accuracy
        accuracy, pair_count = material_pair_accuracy(
            conservative[mask], replay["cost"][mask],
            replay["interaction_group"][mask], args.material_gap,
        )
        by_speed[f"{speed:.1f}"] = {
            "material_pair_accuracy": accuracy,
            "material_pair_count": pair_count,
        }
    return {
        "core": core,
        "heldout_bank": heldout_bank,
        "actor_visited_by_speed": by_speed,
        "flat_train_calibrated_heldout": calibrated,
    }


def save_checkpoint(
    output: Path, total_updates: int,
    critic1, payload1, critic2, payload2, flat_head, flat_payload,
    optimizer1, optimizer2, flat_optimizer,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "total_critic_update_label": total_updates,
        "actor_update_count": 0,
        "optimizer_state_restored": False,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    torch.save({
        **common, "model_class": "AbsoluteActionValueCritic",
        "model": critic1.state_dict(), "training": payload1["training"],
        "optimizer": optimizer1.state_dict(),
    }, output / "critic1.pt")
    torch.save({
        **common, "model_class": "AbsoluteActionValueCritic",
        "model": critic2.state_dict(), "training": payload2["training"],
        "optimizer": optimizer2.state_dict(),
    }, output / "critic2.pt")
    torch.save({
        **common, "model_class": "AbsoluteActionValueCriticFlatHead",
        "model": flat_head.state_dict(),
        "normalization": flat_payload["normalization"],
        "optimizer": flat_optimizer.state_dict(),
    }, output / "flat_head.pt")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checkpoints = [int(value) for value in args.checkpoints.split(",")]
    if checkpoints != [200, 400, 800, 1600, 3200, 6400]:
        raise ValueError(
            "registered extended curve requires checkpoints "
            "200,400,800,1600,3200,6400"
        )
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    parent_contract = json.loads((args.parent_run / "contract.json").read_text())
    parent_summary = json.loads((args.parent_run / "summary.json").read_text())
    outer_fold = int(parent_contract["fold"])
    if outer_fold not in (0, 1, 2):
        raise AssertionError("parent outer fold is not registered")
    if int(parent_summary["actor_update_count"]) != 0:
        raise AssertionError("parent run updated the Actor")
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    train = np.flatnonzero(folds != outer_fold)
    seeds = [int(value) for value in args.seeds.split(",")]
    source_replay_hashes = {
        str(seed): sha256_file(
            args.parent_run / f"seed_{seed}" / "actor_visited_replay.npz"
        ) for seed in seeds
    }
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC1_FIXED_REPLAY_CURVE_CONTRACT",
        "arguments": serialized_args(args),
        "parent_run": str(args.parent_run.resolve()),
        "parent_summary_sha256": sha256_file(args.parent_run / "summary.json"),
        "parent_contract_sha256": sha256_file(args.parent_run / "contract.json"),
        "outer_fold": outer_fold,
        "source_replay_sha256": source_replay_hashes,
        "source_replay_rows_per_seed": 15360,
        "new_dbm_rollouts": 0,
        "actor_module_constructed": False,
        "actor_optimizer_constructed": False,
        "actor_update_count": 0,
        "optimizer_continuation": (
            "parent optimizer state was not serialized; AdamW is reinitialized "
            "once at update label 200, then remains continuous through 6400; "
            "all curve checkpoints now serialize optimizer state"
        ),
        "targets_replay_sampling_and_losses": "unchanged from OAC-1",
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )

    records = []
    for seed in seeds:
        set_seed(26082050 + seed)
        rng = np.random.default_rng(26082050 + seed)
        parent_dir = args.parent_run / f"seed_{seed}"
        with np.load(parent_dir / "actor_visited_replay.npz", allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        if len(replay["cost"]) != 15360:
            raise AssertionError("fixed replay row count mismatch")
        replay_hash_before = sha256_file(parent_dir / "actor_visited_replay.npz")
        critic1, payload1 = load_value_checkpoint(parent_dir / "critic1.pt", device)
        critic2, payload2 = load_value_checkpoint(parent_dir / "critic2.pt", device)
        flat_head, flat_payload = load_flat_checkpoint(parent_dir / "flat_head.pt", device)
        inputs1 = critic_state_inputs(data, payload1)
        inputs2 = critic_state_inputs(data, payload2)
        flat_inputs = critic_state_inputs(data, {
            "training": {"normalization": flat_payload["normalization"]}
        })
        optimizer1 = torch.optim.AdamW(
            critic1.parameters(), lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer2 = torch.optim.AdamW(
            critic2.parameters(), lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
        )
        flat_optimizer = torch.optim.AdamW(
            flat_head.parameters(), lr=args.flat_learning_rate,
            weight_decay=args.weight_decay,
        )
        points = []
        current_total = 200
        evaluation = evaluate_checkpoint(
            args, data, folds, replay, critic1, payload1, critic2, payload2,
            flat_head, flat_payload, device, outer_fold,
        )
        seed_dir = args.output_dir / f"seed_{seed}"
        save_checkpoint(
            seed_dir / "updates_200", 200, critic1, payload1, critic2,
            payload2, flat_head, flat_payload,
            optimizer1, optimizer2, flat_optimizer,
        )
        points.append({"total_updates": 200, "additional_updates": 0, **evaluation})
        for target_total in checkpoints[1:]:
            loss1, loss2, flat_loss = [], [], []
            for _ in range(target_total - current_total):
                training_points = sample_training_points(
                    data, train, replay, 9, args.batch_size, rng
                )
                pairs = sample_pairs(
                    data, train, replay, args.pair_batch_size,
                    args.material_gap, rng,
                )
                loss1.append(update_value_critic(
                    critic1, optimizer1, inputs1, payload1,
                    training_points, pairs, args, device,
                )["loss"])
                loss2.append(update_value_critic(
                    critic2, optimizer2, inputs2, payload2,
                    training_points, pairs, args, device,
                )["loss"])
                flat_loss.append(update_flat_head(
                    flat_head, flat_optimizer, flat_inputs,
                    sample_flat_batch(
                        data, train, replay, args.batch_size,
                        args.flat_gap, rng,
                    ), device,
                ))
            current_total = target_total
            evaluation = evaluate_checkpoint(
                args, data, folds, replay, critic1, payload1, critic2, payload2,
                flat_head, flat_payload, device, outer_fold,
            )
            save_checkpoint(
                seed_dir / f"updates_{target_total}", target_total,
                critic1, payload1, critic2, payload2, flat_head, flat_payload,
                optimizer1, optimizer2, flat_optimizer,
            )
            points.append({
                "total_updates": target_total,
                "additional_updates": target_total - 200,
                "mean_critic1_loss": float(np.mean(loss1)),
                "mean_critic2_loss": float(np.mean(loss2)),
                "mean_flat_loss": float(np.mean(flat_loss)),
                **evaluation,
            })
            correction = evaluation["core"]["bad_action_correction"]
            print(
                f"seed={seed} updates={target_total} "
                f"pair={evaluation['core']['actor_visited']['material_pair_accuracy_conservative']:.3f} "
                f"wrong_fix={correction['initially_wrong_corrected_fraction']:.3f} "
                f"2.8={evaluation['actor_visited_by_speed']['2.8']['material_pair_accuracy']:.3f} "
                f"OOFbank={evaluation['heldout_bank']['material_pair_accuracy']:.3f}",
                flush=True,
            )
        replay_hash_after = sha256_file(parent_dir / "actor_visited_replay.npz")
        if replay_hash_after != replay_hash_before:
            raise AssertionError("source replay changed during fixed-replay curve")
        record = {
            "seed": seed,
            "source_replay": str((parent_dir / "actor_visited_replay.npz").resolve()),
            "source_replay_sha256_before": replay_hash_before,
            "source_replay_sha256_after": replay_hash_after,
            "points": points,
        }
        seed_dir.mkdir(exist_ok=True)
        (seed_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)

    # Mechanism diagnosis is based on the originally registered correction gate.
    correction_at_final = [
        row["points"][-1]["core"]["bad_action_correction"][
            "initially_wrong_corrected_fraction"
        ] for row in records
    ]
    correction_at_1600 = [
        next(
            point for point in row["points"]
            if point["total_updates"] == 1600
        )["core"]["bad_action_correction"][
            "initially_wrong_corrected_fraction"
        ] for row in records
    ]
    oof_bank_delta = [
        row["points"][-1]["heldout_bank"]["material_pair_accuracy"]
        - row["points"][0]["heldout_bank"]["material_pair_accuracy"]
        for row in records
    ]
    further_gain = np.asarray(correction_at_final) - np.asarray(correction_at_1600)
    if sum(value >= 0.02 for value in further_gain) >= 2:
        decision = "EXTENDED_CRITIC_TRAINING_CONTINUES_TO_IMPROVE"
    else:
        decision = "EXTENDED_CRITIC_TRAINING_PLATEAUS_AFTER_1600"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": decision,
        "records": records,
        "final_updates": checkpoints[-1],
        "correction_at_1600": correction_at_1600,
        "correction_at_final": correction_at_final,
        "correction_gain_1600_to_final": further_gain.tolist(),
        "heldout_bank_pair_accuracy_delta_200_to_final": oof_bank_delta,
        "actor_update_count": 0,
        "new_dbm_rollouts": 0,
        "source_replay_unchanged": True,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": decision,
        "correction_at_1600": correction_at_1600,
        "correction_at_final": correction_at_final,
        "correction_gain_1600_to_final": further_gain.tolist(),
        "heldout_bank_delta": oof_bank_delta,
    }, indent=2))


if __name__ == "__main__":
    main()
