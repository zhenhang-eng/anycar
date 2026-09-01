#!/usr/bin/env python3
"""Distill train-only J16 DBM oracles into deterministic 2/6-sigma Actors.

Formal validation J16 actions are never loaded until all Actor training has
finished.  Test episodes are never loaded.  DBM gradients are used only by the
separate offline oracle generator; this trainer consumes knots and forward costs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from train_mppi_direct_center_actor_critic import (
    DEFAULT_PARENT,
    DEFAULT_RISK,
    DEFAULT_SOURCE,
    actor_outputs,
    build_inputs,
    load_direct_partition,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2/"
    "direct_center_actor_trust_selected.pt"
)
DEFAULT_TRAIN_GT = Path(
    "outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2"
)
DEFAULT_VALIDATION_GT = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_center_j16_distillation_20260807_v1"
)
DEFAULT_SCENARIO_PLAN = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1/scenario_plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-gt", type=Path, default=DEFAULT_TRAIN_GT)
    parser.add_argument("--validation-gt", type=Path, default=DEFAULT_VALIDATION_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument(
        "--internal-heldout-selection", action="store_true",
        help=(
            "Train on two of each train speed/scenario triplet, continue all epochs, "
            "and freeze the epoch with lowest third-episode action RMSE."
        ),
    )
    parser.add_argument(
        "--maximum-delta-sigmas", type=float, nargs="+", default=(2.0, 6.0)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gt_knots(
    gt_root: Path,
    expected_split: str,
    label_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary = json.loads((gt_root / "summary.json").read_text())
    if summary["split"] != expected_split:
        raise AssertionError(f"{gt_root} is not split {expected_split}")
    if "test" not in summary["test_policy"]:
        raise AssertionError("GT result does not seal test")
    lookup = {(row["episode"], row["snapshot"]): row for row in summary["rows"]}
    cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    knots, costs = [], []
    for path in label_paths:
        key = (path.parent.name, path.name)
        if key not in cache:
            row = lookup[key]
            with np.load(gt_root / key[0] / key[1], allow_pickle=False) as result:
                index = int(result["knot_best_index"])
                cache[key] = (
                    np.asarray(result["optimized_knots"][index], np.float32),
                    float(row["j16_best_found"]),
                )
        one_knots, one_cost = cache[key]
        knots.append(one_knots)
        costs.append(one_cost)
    return np.asarray(knots, np.float32), np.asarray(costs, np.float32), summary


def target_for_scale(
    oracle_knots: np.ndarray,
    anchor: np.ndarray,
    sigma: np.ndarray,
    maximum_delta_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = maximum_delta_sigma * sigma[:, None, :]
    raw_action = (oracle_knots - anchor) / scale
    reachable = np.all(np.abs(raw_action) <= 1.0 + 1e-6, axis=(1, 2))
    requested = np.clip(raw_action, -1.0, 1.0)
    center = np.clip(anchor + requested * scale, -1.0, 1.0).astype(np.float32)
    effective = ((center - anchor) / scale).astype(np.float32)
    return effective, center, reachable


def state_batch(
    inputs: tuple[np.ndarray, ...], index: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value[index]).to(device) for value in inputs)


def action_fit(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    target: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    action, _ = actor_outputs(actor, inputs, batch_size, device)
    error = action - target
    return {
        "action_rmse": float(np.sqrt(np.mean(error ** 2))),
        "action_mae": float(np.mean(np.abs(error))),
        "action_max_abs_error": float(np.max(np.abs(error))),
        "target_saturation_fraction": float(np.mean(np.abs(target) >= 1.0 - 1e-6)),
    }


def train_one(
    seed: int,
    maximum_delta_sigma: float,
    checkpoint: dict[str, Any],
    inputs: tuple[np.ndarray, ...],
    target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fit_index: np.ndarray | None = None,
    selection_index: np.ndarray | None = None,
) -> tuple[TorchMPPIDeterministicCenterActor, list[dict[str, float]], int]:
    set_seed(seed)
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=maximum_delta_sigma, dropout=0.05
    ).to(device)
    initial_state = copy.deepcopy(checkpoint["actor_state_dict"])
    initial_state.pop("maximum_delta_sigma", None)
    incompatible = actor.load_state_dict(initial_state, strict=False)
    if set(incompatible.missing_keys) != {"maximum_delta_sigma"}:
        raise AssertionError(f"unexpected missing Actor tensors: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise AssertionError(
            f"unexpected source Actor tensors: {incompatible.unexpected_keys}"
        )
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    rng = np.random.default_rng(seed)
    history = []
    fit_index = (
        np.arange(len(target), dtype=np.int64)
        if fit_index is None else np.asarray(fit_index, np.int64)
    )
    selection_inputs = (
        None
        if selection_index is None
        else tuple(value[selection_index] for value in inputs)
    )
    selection_target = None if selection_index is None else target[selection_index]
    best_epoch = args.epochs
    best_selection_rmse = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        actor.train()
        order = rng.permutation(fit_index)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            prediction, _ = actor(*state_batch(inputs, index, device))
            truth = torch.from_numpy(target[index]).to(device)
            loss = F.smooth_l1_loss(prediction, truth, beta=args.huber_beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        actor.eval()
        if selection_inputs is not None and selection_target is not None:
            selection_metrics = action_fit(
                actor,
                selection_inputs,
                selection_target,
                args.evaluation_batch_size,
                device,
            )
            selection_rmse = selection_metrics["action_rmse"]
            if selection_rmse < best_selection_rmse:
                best_selection_rmse = selection_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
        else:
            selection_metrics = {}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **action_fit(actor, inputs, target, args.evaluation_batch_size, device),
                **{
                    f"selection_{name}": value
                    for name, value in selection_metrics.items()
                },
            }
            history.append(row)
            print(
                f"[scale={maximum_delta_sigma:g} seed={seed} epoch={epoch:03d}] "
                f"loss={row['train_loss']:.5f} rmse={row['action_rmse']:.5f} "
                f"lr={row['learning_rate']:.2g}",
                flush=True,
            )
    if best_state is not None:
        actor.load_state_dict(best_state, strict=True)
    actor.eval()
    return actor, history, best_epoch


def internal_episode_indices(
    plan_path: Path, label_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    plan = json.loads(plan_path.read_text())
    groups: dict[tuple[float, str], list[str]] = {}
    for row in plan["episodes"]:
        if row["split"] != "train":
            continue
        key = (float(row["reference_speed_mps"]), str(row["scenario_class"]))
        groups.setdefault(key, []).append(str(row["episode_id"]))
    fit_episodes, heldout_episodes = [], []
    for key, episodes in sorted(groups.items()):
        episodes.sort()
        if len(episodes) != 3:
            raise AssertionError(f"train stratum {key} does not contain 3 episodes")
        fit_episodes.extend(episodes[:2])
        heldout_episodes.append(episodes[2])
    fit_set, heldout_set = set(fit_episodes), set(heldout_episodes)
    context_episodes = np.asarray([path.parent.name for path in label_paths])
    fit_index = np.flatnonzero(np.isin(context_episodes, list(fit_set)))
    heldout_index = np.flatnonzero(np.isin(context_episodes, list(heldout_set)))
    if len(fit_index) + len(heldout_index) != len(label_paths):
        raise AssertionError("internal episode split does not cover train contexts")
    return fit_index, heldout_index, fit_episodes, heldout_episodes


def evaluate_center_methods(
    methods: dict[str, np.ndarray],
    label_paths: list[Path],
    source_root: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    names = tuple(methods)
    if any(len(value) != len(label_paths) for value in methods.values()):
        raise ValueError("method/context count mismatch")
    output = {name: np.empty(len(label_paths), np.float32) for name in names}
    cursor = 0
    while cursor < len(label_paths):
        path = label_paths[cursor]
        end = cursor
        while end < len(label_paths) and label_paths[end] == path:
            end += 1
        source_path = source_root / path.parent.name / "snapshots" / path.name
        centers = np.concatenate(
            [methods[name][cursor:end] for name in names], axis=0
        ).astype(np.float32)
        with np.load(source_path, allow_pickle=False) as source:
            config = {
                "objective": {
                    "cost_weights": json.loads(str(source["cost_weights_json"]))
                }
            }
            controller, backend = make_controller(source, config, device)
            cost, _, _ = evaluate_knots(
                controller,
                backend,
                centers,
                torch.from_numpy(source["history"]).to(device),
                torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                controller._prepare_reference(source["reference"]),
            )
        count = end - cursor
        cost = np.asarray(cost, np.float32).reshape(len(names), count)
        for method_index, name in enumerate(names):
            output[name][cursor:end] = cost[method_index]
        cursor = end
    return output


def distribution(cost: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(cost)),
        "median": float(np.median(cost)),
        "p95": float(np.quantile(cost, 0.95)),
        "maximum": float(np.max(cost)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    splits = json.loads((args.labels / "splits.json").read_text())
    if not bool(splits.get("test_sealed_not_generated", False)):
        raise AssertionError("split manifest does not seal the ungenerated test split")
    if set(splits["train"]) & set(splits["validation"]):
        raise AssertionError("invalid episode split")

    def load_split(split: str, gt_root: Path) -> dict[str, Any]:
        episodes = splits[split]
        state = load_state_partition(
            args.source, args.parent_labels, episodes, "selection"
        )
        replay = load_direct_partition(
            args.labels,
            args.parent_labels,
            args.risk_labels,
            episodes,
            float(checkpoint["maximum_delta_sigma"]),
        )
        inputs = build_inputs(state, replay, checkpoint)
        oracle_knots, j16_cost, gt_summary = load_gt_knots(
            gt_root, split, replay.label_paths
        )
        return {
            "state": state,
            "replay": replay,
            "inputs": inputs,
            "oracle_knots": oracle_knots,
            "j16_cost": j16_cost,
            "gt_summary": gt_summary,
        }

    # Only train episodes and train-oracle labels are visible during optimization.
    partitions = {"train": load_split("train", args.train_gt)}
    trained: dict[str, TorchMPPIDeterministicCenterActor] = {}
    training_rows = []
    targets: dict[str, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "train": {}
    }
    if args.internal_heldout_selection:
        fit_index, selection_index, fit_episodes, selection_episodes = (
            internal_episode_indices(
                args.scenario_plan, partitions["train"]["replay"].label_paths
            )
        )
    else:
        fit_index = selection_index = None
        fit_episodes = list(splits["train"])
        selection_episodes = []
    for scale in args.maximum_delta_sigmas:
        part = partitions["train"]
        targets["train"][scale] = target_for_scale(
            part["oracle_knots"],
            part["state"].anchor,
            part["replay"].sigma,
            scale,
        )
        train_target = targets["train"][scale][0]
        for seed in args.seeds:
            actor, history, best_epoch = train_one(
                seed,
                scale,
                checkpoint,
                partitions["train"]["inputs"],
                train_target,
                args,
                device,
                fit_index,
                selection_index,
            )
            key = f"actor_scale{scale:g}_seed{seed}"
            trained[key] = copy.deepcopy(actor).cpu()
            torch.save(
                {
                    "format_version": 1,
                    "method": "train-only J16 projected-target distillation",
                    "qualification": "VALIDATION_ONLY_TEST_SEALED",
                    "maximum_delta_sigma": scale,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "internal_fit_episodes": fit_episodes,
                    "internal_selection_episodes": selection_episodes,
                    "actor_state_dict": trained[key].state_dict(),
                    "source_checkpoint": str(args.checkpoint),
                    "source_checkpoint_sha256": sha256_file(args.checkpoint),
                    "state_normalization": checkpoint["state_normalization"],
                    "feedback_mean": checkpoint["feedback_mean"],
                    "feedback_std": checkpoint["feedback_std"],
                    "gradient_mean": checkpoint["gradient_mean"],
                    "gradient_std": checkpoint["gradient_std"],
                    "training_history": history,
                    "test_policy": "test split not loaded or evaluated",
                },
                args.output_dir / f"{key}.pt",
            )
            training_rows.append({
                "method": key,
                "selected_epoch": best_epoch,
                "internal_fit_episodes": fit_episodes,
                "internal_selection_episodes": selection_episodes,
                "history": history,
            })

    # Formal validation oracle actions are loaded only after every Actor is frozen.
    partitions["validation"] = load_split("validation", args.validation_gt)
    targets["validation"] = {}
    for scale in args.maximum_delta_sigmas:
        part = partitions["validation"]
        targets["validation"][scale] = target_for_scale(
            part["oracle_knots"],
            part["state"].anchor,
            part["replay"].sigma,
            scale,
        )

    summary: dict[str, Any] = {
        "format_version": 1,
        "method": "train-only J16 projected-target deterministic Actor distillation",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "train_gt": str(args.train_gt),
        "validation_gt": str(args.validation_gt),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "maximum_delta_sigmas": args.maximum_delta_sigmas,
        "internal_heldout_selection": args.internal_heldout_selection,
        "internal_fit_episodes": fit_episodes,
        "internal_selection_episodes": selection_episodes,
        "training": training_rows,
        "splits": {},
        "test_policy": "test split not loaded or evaluated",
    }
    for split in ("train", "validation"):
        part = partitions[split]
        methods = {}
        for scale in args.maximum_delta_sigmas:
            methods[f"projected_j16_scale{scale:g}"] = targets[split][scale][1]
        for name, actor_cpu in trained.items():
            actor = actor_cpu.to(device)
            _, centers = actor_outputs(
                actor,
                part["inputs"],
                args.evaluation_batch_size,
                device,
            )
            methods[name] = centers
            actor_cpu.cpu()
        direct = evaluate_center_methods(
            methods, part["replay"].label_paths, args.source, device
        )
        split_summary = {
            "episodes": splits[split],
            "snapshots": int(part["gt_summary"]["snapshot_count"]),
            "contexts": len(part["replay"].label_paths),
            "j16": distribution(part["j16_cost"]),
            "methods": {},
        }
        for scale in args.maximum_delta_sigmas:
            action, _, reachable = targets[split][scale]
            split_summary[f"scale{scale:g}_target"] = {
                "reachable_fraction": float(np.mean(reachable)),
                "target_saturation_fraction": float(
                    np.mean(np.abs(action) >= 1.0 - 1e-6)
                ),
            }
        for name, cost in direct.items():
            record = {
                **distribution(cost),
                "gap_to_j16_mean": float(np.mean(cost - part["j16_cost"])),
            }
            if name.startswith("actor_scale"):
                scale = float(name.split("_seed")[0].replace("actor_scale", ""))
                record.update(action_fit(
                    trained[name].to(device),
                    part["inputs"],
                    targets[split][scale][0],
                    args.evaluation_batch_size,
                    device,
                ))
                trained[name].cpu()
            split_summary["methods"][name] = record
        summary["splits"][split] = split_summary
        print(json.dumps({split: split_summary}, indent=2), flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
