#!/usr/bin/env python3
"""Cross-fit one/two-stage Actor refinement on strong-search incumbent paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    TorchMPPISemanticStateActionEncoder,
)
from generate_dbm_direct_gt_validation import interpolate_knots
from pretrain_highspeed_actor_twin_critic import load_data, make_folds


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1"
)
DEFAULT_STRONG = Path(
    "outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_iterative_path_distillation_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32).reshape(1, 1, 2)
TRAINING_SEEDS = (0, 1, 2)
ARMS = ("one_step", "direct_two_step", "stage2", "cascade_two_step")
LEARNED_ARMS = ("one_step", "direct_two_step", "stage2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--strong-dir", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def build_actor(maximum_delta_sigma: float) -> TorchMPPIDeterministicCenterActor:
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=maximum_delta_sigma, dropout=0.0
    )
    actor.encoder = TorchMPPISemanticStateActionEncoder(
        include_feedback=False, dropout=0.0
    )
    return actor


def normalize_inputs(
    data: dict[str, np.ndarray], fit_states: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], MPPIProposalNormalization]:
    normalizer = MPPIProposalNormalization.fit(
        data["history"][fit_states], data["reference"][fit_states],
        data["current"][fit_states],
    )
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32),
        current.astype(np.float32), np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    ), normalizer


def actor_forward(
    actor: torch.nn.Module, inputs: tuple[np.ndarray, ...], state_rows: np.ndarray,
    anchors: np.ndarray, device: torch.device, batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    actions, centers = [], []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(state_rows), batch_size):
            local = slice(start, start + batch_size)
            rows = state_rows[local]
            history = torch.from_numpy(inputs[0][rows]).to(device)
            reference = torch.from_numpy(inputs[1][rows]).to(device)
            current = torch.from_numpy(inputs[2][rows]).to(device)
            anchor = torch.from_numpy(anchors[local]).to(device)
            feedback = torch.from_numpy(inputs[3][rows]).to(device)
            gradient = torch.from_numpy(inputs[4][rows]).to(device)
            action, center = actor(
                history, reference, current, anchor, feedback, gradient
            )
            actions.append(action.cpu().numpy()); centers.append(center.cpu().numpy())
    return np.concatenate(actions).astype(np.float32), np.concatenate(centers).astype(np.float32)


def train_actor(
    arm: str, maximum_delta_sigma: float, anchors: np.ndarray,
    targets: np.ndarray, example_states: np.ndarray, fit_examples: np.ndarray,
    selection_examples: np.ndarray, inputs: tuple[np.ndarray, ...],
    normalizer: MPPIProposalNormalization, fold: int, seed: int,
    args: argparse.Namespace, output: Path,
) -> tuple[TorchMPPIDeterministicCenterActor, dict[str, Any], Path]:
    torch.manual_seed(62_000 + fold * 100 + seed * 10 + LEARNED_ARMS.index(arm))
    np.random.seed(62_000 + fold * 100 + seed * 10 + LEARNED_ARMS.index(arm))
    actor = build_actor(maximum_delta_sigma).to(args.device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-5)
    target_action = (
        (targets - anchors) / (maximum_delta_sigma * SIGMA)
    ).astype(np.float32)
    if np.max(np.abs(target_action)) > 1.00001:
        raise AssertionError(f"{arm} target exceeds its registered trust box")
    rng = np.random.default_rng(72_000 + fold * 100 + seed * 10 + LEARNED_ARMS.index(arm))
    best_state, best_epoch, best_loss = None, 0, math.inf
    patience = args.patience
    for epoch in range(1, args.epochs + 1):
        actor.train()
        order = rng.permutation(fit_examples)
        for start in range(0, len(order), args.batch_size):
            examples = order[start : start + args.batch_size]
            states = example_states[examples]
            history = torch.from_numpy(inputs[0][states]).to(args.device)
            reference = torch.from_numpy(inputs[1][states]).to(args.device)
            current = torch.from_numpy(inputs[2][states]).to(args.device)
            anchor = torch.from_numpy(anchors[examples]).to(args.device)
            feedback = torch.from_numpy(inputs[3][states]).to(args.device)
            gradient = torch.from_numpy(inputs[4][states]).to(args.device)
            target = torch.from_numpy(target_action[examples]).to(args.device)
            predicted, _ = actor(
                history, reference, current, anchor, feedback, gradient
            )
            loss = torch.mean((predicted - target) ** 2)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
        predicted, _ = actor_forward(
            actor, inputs, example_states[selection_examples],
            anchors[selection_examples], torch.device(args.device),
        )
        selection_loss = float(np.mean(
            (predicted - target_action[selection_examples]) ** 2
        ))
        if selection_loss < best_loss - 1e-7:
            best_loss = selection_loss; best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict()); patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    actor.load_state_dict(best_state, strict=True)
    fit_prediction, _ = actor_forward(
        actor, inputs, example_states[fit_examples], anchors[fit_examples],
        torch.device(args.device),
    )
    fit_mse = float(np.mean((fit_prediction - target_action[fit_examples]) ** 2))
    checkpoint = output / f"fold_{fold}" / f"seed_{seed}" / f"{arm}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "highspeed_iterative_path_actor_v1",
        "arm": arm, "fold": fold, "seed": seed,
        "maximum_delta_sigma": maximum_delta_sigma,
        "model_state_dict": actor.state_dict(),
        "normalizer": normalizer.to_dict(),
        "fit_example_indices": fit_examples,
        "selection_example_indices": selection_examples,
        "best_epoch": best_epoch, "selection_mse": best_loss,
        "fit_mse": fit_mse,
    }
    torch.save(payload, checkpoint)
    return actor, {
        "best_epoch": best_epoch, "selection_mse": best_loss,
        "fit_mse": fit_mse, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
    }, checkpoint


def evaluate_components(
    data: dict[str, np.ndarray], state_rows: np.ndarray, centers: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(state_rows)
    actions = interpolate_knots(torch.from_numpy(centers).to(device), params.horizon)
    initial = torch.from_numpy(data["state_six"][state_rows]).to(device)
    with torch.no_grad():
        full = backend.rollout_full_state_differentiable(initial, actions)
    trajectory = full[..., [0, 1, 2, 3, 5]]
    reference = torch.from_numpy(data["rollout_reference"][state_rows]).to(device)
    current = torch.from_numpy(data["current_action"][state_rows]).to(device)
    error = trajectory[..., :2] - reference[..., :2]
    yaw = reference[..., 2]
    along = error[..., 0] * torch.cos(yaw) + error[..., 1] * torch.sin(yaw)
    cross = -error[..., 0] * torch.sin(yaw) + error[..., 1] * torch.cos(yaw)
    yaw_delta = torch.atan2(
        torch.sin(trajectory[..., 2] - yaw), torch.cos(trajectory[..., 2] - yaw)
    )
    previous = torch.cat((current[:, None], actions[:, :-1]), dim=1)
    rate = actions - previous
    parts_t = {
        "position_along": weights.position * along.square().sum(-1),
        "position_cross": weights.position * cross.square().sum(-1),
        "yaw": weights.yaw * yaw_delta.square().sum(-1),
        "vx": weights.vx * (trajectory[..., 3] - reference[..., 3]).square().sum(-1),
        "acceleration_rate": weights.acceleration_rate * rate[..., 0].square().sum(-1),
        "steering_rate": weights.steering_rate * rate[..., 1].square().sum(-1),
    }
    parts = {key: value.cpu().numpy().astype(np.float32) for key, value in parts_t.items()}
    total = sum(parts.values())
    return total.astype(np.float32), parts


def proposal_metrics(
    cost: np.ndarray, start_cost: np.ndarray, target_cost: np.ndarray,
    strong_cost: np.ndarray, parts: dict[str, np.ndarray] | None = None,
    start_parts: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    gain = start_cost - cost
    target_gain = start_cost - target_cost
    strong_gain = start_cost - strong_cost
    result: dict[str, Any] = {
        "cost": distribution(cost), "start_relative_gain": distribution(gain),
        "strictly_beats_start_fraction": float(np.mean(gain > 1e-5)),
        "regression_fraction": float(np.mean(gain < -1e-5)),
        "target_gain_recovery": float(gain.sum() / target_gain.sum()),
        "strong_headroom_recovery": float(gain.sum() / strong_gain.sum()),
    }
    if parts is not None and start_parts is not None:
        total_gain = float(gain.sum())
        result["component_gain_fraction"] = {
            key: float((start_parts[key] - parts[key]).sum() / total_gain)
            for key in parts
        } if abs(total_gain) > 1e-12 else {}
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    strong_path = (args.strong_dir / "oracle.npz").resolve()
    strong_summary_path = (args.strong_dir / "summary.json").resolve()
    strong_validator_path = (args.strong_dir / "validator_report.json").resolve()
    strong_summary = json.loads(strong_summary_path.read_text())
    strong_validator = json.loads(strong_validator_path.read_text())
    if strong_validator["qualification"] != "HIGHSPEED_STRONG_SEARCH_ORACLE_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("strong-search source is not independently validated")
    if strong_summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("strong-search source is not train-only")
    with np.load(strong_path, allow_pickle=False) as loaded:
        strong = {name: np.asarray(loaded[name]) for name in loaded.files}
    full = load_data(args.replay_dir, args.teacher_dir)
    source_rows = strong["source_indices"].astype(np.int64)
    data = {
        key: (value[source_rows] if isinstance(value, np.ndarray) and value.ndim
              and len(value) == len(full["episode"]) else value)
        for key, value in full.items()
    }
    if not np.array_equal(data["episode"], strong["episode_id"]):
        raise AssertionError("strong path/data alignment failed")
    path_centers = strong["path_centers"][:, 2:5].astype(np.float32)
    path_costs = strong["path_costs"][:, 2:5].astype(np.float32)
    state_count, source_seeds = path_centers.shape[:2]
    if (state_count, source_seeds) != (120, 3):
        raise AssertionError("unexpected Actor path bank shape")
    example_states = np.repeat(np.arange(state_count, dtype=np.int64), source_seeds)
    start = path_centers[:, :, 0].reshape(-1, 8, 2)
    path1 = path_centers[:, :, 1].reshape(-1, 8, 2)
    path2 = path_centers[:, :, 2].reshape(-1, 8, 2)
    start_cost = path_costs[:, :, 0].reshape(-1)
    path1_cost = path_costs[:, :, 1].reshape(-1)
    path2_cost = path_costs[:, :, 2].reshape(-1)
    strong_cost = np.repeat(strong["oracle_costs"], source_seeds)
    warm_cost = np.repeat(strong["start_costs"][:, 0], source_seeds)
    folds = make_folds(data["episode"], data["speed"], data["scenario"])
    if [int(np.sum(folds == fold)) for fold in range(5)] != [24] * 5:
        raise AssertionError("expected 24 physical states per outer fold")
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    output.mkdir(parents=True)

    # Baselines and label paths are evaluated once for exact component attribution.
    base_states = example_states
    _, start_parts = evaluate_components(data, base_states, start, backend, weights, params, device)
    label1_replay, label1_parts = evaluate_components(
        data, base_states, path1, backend, weights, params, device
    )
    label2_replay, label2_parts = evaluate_components(
        data, base_states, path2, backend, weights, params, device
    )
    if np.max(np.abs(label1_replay - path1_cost)) > 0.1 or np.max(np.abs(label2_replay - path2_cost)) > 0.1:
        raise AssertionError("path label DBM replay mismatch")

    oof_centers = {
        arm: np.full((len(TRAINING_SEEDS), len(start), 8, 2), np.nan, np.float32)
        for arm in ARMS
    }
    records = []
    for fold in range(5):
        selection_fold = (fold + 1) % 5
        fit_states = np.flatnonzero((folds != fold) & (folds != selection_fold))
        selection_states = np.flatnonzero(folds == selection_fold)
        oof_states = np.flatnonzero(folds == fold)
        fit_examples = np.flatnonzero(np.isin(example_states, fit_states))
        selection_examples = np.flatnonzero(np.isin(example_states, selection_states))
        oof_examples = np.flatnonzero(np.isin(example_states, oof_states))
        inputs, normalizer = normalize_inputs(data, fit_states)
        for seed in TRAINING_SEEDS:
            learned: dict[str, TorchMPPIDeterministicCenterActor] = {}
            training: dict[str, Any] = {}
            for arm, maximum, anchors, targets in (
                ("one_step", 1.0, start, path1),
                ("direct_two_step", 2.0, start, path2),
                # The second search ring has 0.7-sigma RMS radius, while the
                # deterministic rotated basis reaches 0.981 sigma in an
                # individual coordinate.  A 1.0-sigma component box preserves
                # every recorded transition without clipping the label.
                ("stage2", 1.0, path1, path2),
            ):
                model, report, _ = train_actor(
                    arm, maximum, anchors, targets, example_states,
                    fit_examples, selection_examples, inputs, normalizer,
                    fold, seed, args, output,
                )
                learned[arm] = model; training[arm] = report
            oof_state_rows = example_states[oof_examples]
            _, one = actor_forward(
                learned["one_step"], inputs, oof_state_rows, start[oof_examples], device
            )
            _, direct2 = actor_forward(
                learned["direct_two_step"], inputs, oof_state_rows,
                start[oof_examples], device,
            )
            _, stage2_true = actor_forward(
                learned["stage2"], inputs, oof_state_rows, path1[oof_examples], device
            )
            _, cascade = actor_forward(
                learned["stage2"], inputs, oof_state_rows, one, device
            )
            for arm, value in (
                ("one_step", one), ("direct_two_step", direct2),
                ("stage2", stage2_true), ("cascade_two_step", cascade),
            ):
                oof_centers[arm][seed, oof_examples] = value
            records.append({
                "fold": fold, "seed": seed,
                "fit_physical_states": int(len(fit_states)),
                "selection_physical_states": int(len(selection_states)),
                "oof_physical_states": int(len(oof_states)),
                "fit_episodes": sorted(data["episode"][fit_states].astype(str).tolist()),
                "selection_episodes": sorted(data["episode"][selection_states].astype(str).tolist()),
                "oof_episodes": sorted(data["episode"][oof_states].astype(str).tolist()),
                "training": training,
            })
            print(json.dumps({
                "fold": fold, "seed": seed,
                "epochs": {key: value["best_epoch"] for key, value in training.items()},
                "selection_mse": {key: value["selection_mse"] for key, value in training.items()},
            }), flush=True)

    if any(not np.isfinite(value).all() for value in oof_centers.values()):
        raise AssertionError("OOF predictions are incomplete")
    label_metrics = {
        "path1": proposal_metrics(path1_cost, start_cost, path1_cost, strong_cost,
                                  label1_parts, start_parts),
        "path2": proposal_metrics(path2_cost, start_cost, path2_cost, strong_cost,
                                  label2_parts, start_parts),
    }
    per_seed: dict[str, Any] = {}
    oof_costs: dict[str, np.ndarray] = {
        arm: np.empty((len(TRAINING_SEEDS), len(start)), np.float32) for arm in ARMS
    }
    oof_parts: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        oof_parts[arm] = {
            key: np.empty((len(TRAINING_SEEDS), len(start)), np.float32)
            for key in start_parts
        }
    for seed in TRAINING_SEEDS:
        seed_report: dict[str, Any] = {}
        for arm in ARMS:
            cost, parts = evaluate_components(
                data, example_states, oof_centers[arm][seed],
                backend, weights, params, device,
            )
            oof_costs[arm][seed] = cost
            for key in parts:
                oof_parts[arm][key][seed] = parts[key]
            target_cost = path1_cost if arm == "one_step" else path2_cost
            seed_report[arm] = proposal_metrics(
                cost, start_cost, target_cost, strong_cost, parts, start_parts
            )
            guarded = np.minimum(cost, warm_cost)
            seed_report[arm]["warm_guarded"] = {
                "cost": distribution(guarded),
                "warm_relative_reduction": float((warm_cost - guarded).sum() / warm_cost.sum()),
                "warm_regression_fraction": 0.0,
            }
        # Isolate stage-2 transfer at its true search anchor.
        stage2_gain = path1_cost - oof_costs["stage2"][seed]
        true_stage2_gain = path1_cost - path2_cost
        seed_report["stage2"]["transition_gain_recovery"] = float(
            stage2_gain.sum() / true_stage2_gain.sum()
        )
        per_seed[str(seed)] = seed_report

    gates = {}
    for arm in ("one_step", "direct_two_step", "cascade_two_step"):
        transfer = np.asarray([
            per_seed[str(seed)][arm]["target_gain_recovery"] for seed in TRAINING_SEEDS
        ])
        p05 = np.asarray([
            per_seed[str(seed)][arm]["start_relative_gain"]["p05"]
            for seed in TRAINING_SEEDS
        ])
        gates[arm] = {
            "target_gain_recovery_by_seed": transfer.tolist(),
            "gain_p05_by_seed": p05.tolist(),
            "mechanism_at_least_2_of_3_recovery_ge_0_50": bool(np.sum(transfer >= 0.50) >= 2),
            "raw_tail_at_least_2_of_3_p05_nonnegative": bool(np.sum(p05 >= 0.0) >= 2),
        }

    archive_path = output / "oof_predictions.npz"
    archive: dict[str, np.ndarray] = {
        "source_indices": source_rows, "episode_id": data["episode"],
        "scenario_class": data["scenario"], "speed_kph": data["speed"],
        "fold": folds, "example_state_indices": example_states,
        "source_actor_seed": np.tile(np.arange(3, dtype=np.int64), state_count),
        "start_centers": start, "path1_centers": path1, "path2_centers": path2,
        "start_cost": start_cost, "path1_cost": path1_cost, "path2_cost": path2_cost,
        "strong_cost": strong_cost, "warm_cost": warm_cost,
    }
    for arm in ARMS:
        archive[f"{arm}_centers"] = oof_centers[arm]
        archive[f"{arm}_cost"] = oof_costs[arm]
        for key, value in oof_parts[arm].items():
            archive[f"{arm}_{key}"] = value
    np.savez_compressed(archive_path, **archive)
    summary = {
        "format": "highspeed_iterative_path_distillation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ITERATIVE_PATH_DISTILLATION_PENDING_INDEPENDENT_VALIDATION",
        "contract": {
            "split": "train-only; 120 independent step-0 physical states",
            "examples": int(len(start)), "source_actor_paths_per_state": 3,
            "folds": 5, "training_seeds": list(TRAINING_SEEDS),
            "inner_selection_fold": "(outer_fold + 1) mod 5; remaining three folds fit",
            "architecture": "semantic-clean explicit-anchor deterministic residual Actor",
            "arms": {
                "one_step": "start->path1, 1.0 sigma bound",
                "direct_two_step": "start->path2, 2.0 sigma bound",
                "stage2": "true path1->path2, 0.7 sigma RMS ring / 1.0 sigma component bound",
                "cascade_two_step": "learned stage1 then learned stage2",
            },
            "objective": "action-space transition MSE; DBM only for evaluation",
            "formal_validation_or_test_opened": False,
        },
        "path_audit": {
            "path1_strong_headroom_recovery": label_metrics["path1"]["strong_headroom_recovery"],
            "path2_strong_headroom_recovery": label_metrics["path2"]["strong_headroom_recovery"],
            "path1_increment_rms_sigma": distribution(np.sqrt(np.mean(
                ((path1 - start) / SIGMA) ** 2, axis=(-1, -2)
            ))),
            "path2_increment_from_path1_rms_sigma": distribution(np.sqrt(np.mean(
                ((path2 - path1) / SIGMA) ** 2, axis=(-1, -2)
            ))),
        },
        "sources": {
            "strong_oracle": str(strong_path), "strong_oracle_sha256": sha256(strong_path),
            "strong_summary": str(strong_summary_path),
            "strong_summary_sha256": sha256(strong_summary_path),
            "strong_validator": str(strong_validator_path),
            "strong_validator_sha256": sha256(strong_validator_path),
            "replay": str((args.replay_dir / "replay.npz").resolve()),
            "replay_sha256": str(full["replay_sha256"]),
            "teacher": str((args.teacher_dir / "labels.npz").resolve()),
            "teacher_sha256": str(full["teacher_sha256"]),
        },
        "training": {
            "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "lr": args.lr,
        },
        "label_metrics": label_metrics,
        "per_training_seed": per_seed,
        "gates": gates,
        "records": records,
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256(archive_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"gates": gates, "per_training_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()
