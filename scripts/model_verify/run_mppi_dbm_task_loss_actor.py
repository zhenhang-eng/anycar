#!/usr/bin/env python3
"""Direct differentiable-DBM task-loss capacity diagnostic for the G-X Actor.

The scalar loss is the real deterministic J50 produced by the frozen DBM and
the repository MPPI cost.  No Critic, teacher MSE, TD target or finite-
difference gradient participates in the update.  The script first overfits 32
and 128 train-only states.  It runs a complete episode-grouped fold-0 training
split only when both registered tiny gates pass.  Formal validation/test stay
sealed.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from run_mppi_absolute_action_value_critic_cv import make_folds
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_multi_candidate_actor import MultiCandidateGTXActor, distribution


DEFAULT_BANK = Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1")
DEFAULT_ACTOR = Path("outputs/mppi_proposal/j16_multi_candidate_actor_20260824_v1")
DEFAULT_GT_V1 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/dbm_task_loss_actor_20260824_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny-sizes", default="32,128")
    parser.add_argument("--tiny-updates", type=int, default=1200)
    parser.add_argument("--full-updates", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subset(data: dict[str, np.ndarray], pool: np.ndarray,
                      count: int, seed: int) -> np.ndarray:
    cells: dict[tuple[float, str], list[int]] = {}
    for index in pool:
        key = (round(float(data["speed"][index]), 2), str(data["scenario"][index]))
        cells.setdefault(key, []).append(int(index))
    rng = np.random.default_rng(seed)
    for values in cells.values():
        rng.shuffle(values)
    keys = sorted(cells)
    result = []
    offset = 0
    while len(result) < count:
        key = keys[offset % len(keys)]
        values = cells[key]
        take = offset // len(keys)
        if take < len(values):
            result.append(values[take])
        offset += 1
    return np.asarray(result, np.int64)


def make_inputs(data: dict[str, np.ndarray], normalizer: MPPIProposalNormalization,
                device: torch.device) -> tuple[torch.Tensor, ...]:
    values = normalizer.normalize_numpy(data["history"], data["reference"], data["current"])
    return tuple(torch.from_numpy(value.astype(np.float32)).to(device) for value in values)


def load_actor(path: Path, device: torch.device) -> tuple[MultiCandidateGTXActor, dict]:
    payload = torch.load(path, map_location=device)
    if int(payload["candidate_count"]) != 1:
        raise AssertionError("task-loss pilot requires the paired K=1 checkpoint")
    state = payload["model_state_dict"]
    model = MultiCandidateGTXActor(
        1, state["out_center"].reshape(8, 2), state["out_scale"].reshape(8, 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model, payload


def rollout_cost(backend, weights, params, knots, states, current, reference,
                 indices, batch_size, device, differentiable=False) -> torch.Tensor:
    values = []
    for begin in range(0, len(indices), batch_size):
        local = indices[begin:begin + batch_size]
        action = interpolate_knots(knots[local], params.horizon).unsqueeze(1)
        value = batched_cost(
            backend, weights, action,
            torch.from_numpy(states[local]).to(device),
            torch.from_numpy(current[local]).to(device),
            torch.from_numpy(reference[local]).to(device),
        )[:, 0]
        values.append(value if differentiable else value.detach())
    return torch.cat(values)


def predict(model: MultiCandidateGTXActor, inputs: tuple[torch.Tensor, ...],
            indices: np.ndarray, batch_size: int = 256) -> torch.Tensor:
    values = []
    for begin in range(0, len(indices), batch_size):
        local = torch.from_numpy(indices[begin:begin + batch_size]).to(inputs[0].device)
        values.append(model(*(value[local] for value in inputs))[:, 0])
    return torch.cat(values)


def recovery(initial: np.ndarray, oracle: np.ndarray, final: np.ndarray) -> float:
    return float(np.sum(initial - final) / max(float(np.sum(initial - oracle)), 1e-9))


def evaluate_model(model, inputs, backend, weights, params, states, current,
                   reference, indices, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        action = predict(model, inputs, indices)
        local_knots = torch.empty(
            len(states), 8, 2, device=device, dtype=action.dtype
        )
        local_knots[torch.from_numpy(indices).to(device)] = action
        cost = rollout_cost(
            backend, weights, params, local_knots, states, current, reference,
            indices, 128, device,
        )
    return action.cpu().numpy().astype(np.float32), cost.cpu().numpy().astype(np.float32)


def train_task_loss(
    base_model, inputs, backend, weights, params, states, current, reference,
    train_indices, updates, batch_size, learning_rate, weight_decay,
    evaluation_interval, device, seed,
) -> tuple[MultiCandidateGTXActor, list[dict[str, float]]]:
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        _, initial_cost = evaluate_model(
            model, inputs, backend, weights, params, states, current, reference,
            train_indices, device,
        )
    cost_scale = max(float(np.mean(initial_cost)), 1.0)
    best_state = copy.deepcopy(model.state_dict())
    best_mean = float(np.mean(initial_cost))
    trace = [{"update": 0, "mean_cost": best_mean, "median_cost": float(np.median(initial_cost))}]
    for update in range(1, updates + 1):
        index = rng.choice(
            train_indices, size=min(batch_size, len(train_indices)), replace=False
        ).astype(np.int64)
        index_t = torch.from_numpy(index).to(device)
        model.train()
        knots = model(*(value[index_t] for value in inputs))[:, 0]
        actions = interpolate_knots(knots, params.horizon).unsqueeze(1)
        cost = batched_cost(
            backend, weights, actions,
            torch.from_numpy(states[index]).to(device),
            torch.from_numpy(current[index]).to(device),
            torch.from_numpy(reference[index]).to(device),
        )[:, 0]
        # Positive scalar normalization leaves the exact mean-J minimizer unchanged.
        loss = cost.mean() / cost_scale
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if update % evaluation_interval == 0 or update == updates:
            _, evaluated = evaluate_model(
                model, inputs, backend, weights, params, states, current,
                reference, train_indices, device,
            )
            mean_value = float(np.mean(evaluated))
            trace.append({
                "update": update, "mean_cost": mean_value,
                "median_cost": float(np.median(evaluated)),
            })
            if mean_value < best_mean:
                best_mean = mean_value
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, trace


def result_metrics(initial: np.ndarray, oracle: np.ndarray,
                   final: np.ndarray) -> dict[str, Any]:
    gain = initial - final
    return {
        "initial_cost": distribution(initial), "j16_cost": distribution(oracle),
        "final_cost": distribution(final), "gain": distribution(gain),
        "headroom_recovery": recovery(initial, oracle, final),
        "improved_fraction": float(np.mean(gain > 1e-5)),
        "regressed_fraction": float(np.mean(gain < -1e-5)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    with np.load(args.bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    folds = make_folds(data, 3)
    train_pool = np.flatnonzero(folds != args.fold)
    heldout = np.flatnonzero(folds == args.fold)
    checkpoint_path = args.actor_root / f"actor_k1_fold{args.fold}_seed{args.seed}.pt"
    base_model, payload = load_actor(checkpoint_path, device)
    normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
    inputs = make_inputs(data, normalizer, device)
    states, current_action, direct_reference, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    best_index = np.argmin(data["costs"][:, 16:24], axis=1)
    j16_knots = data["actions"][:, 16:24][np.arange(len(data["actions"])), best_index]
    j16_cost = data["costs"][:, 16:24][np.arange(len(data["costs"])), best_index]

    # Output-support oracle: project J16 into the exact current G-X affine-tanh box.
    center = base_model.out_center.detach().cpu().numpy().reshape(8, 2)
    scale = base_model.out_scale.detach().cpu().numpy().reshape(8, 2)
    projected = np.clip(j16_knots, center - scale, center + scale).astype(np.float32)
    projected_all = torch.from_numpy(projected).to(device)
    projected_cost = rollout_cost(
        backend, weights, params, projected_all, states, current_action,
        direct_reference, np.arange(len(states)), 128, device,
    ).cpu().numpy()
    outside = (j16_knots < center - scale) | (j16_knots > center + scale)

    tiny_results: dict[str, Any] = {}
    tiny_passes = []
    for size in [int(x) for x in args.tiny_sizes.split(",")]:
        subset = stratified_subset(data, train_pool, size, 260824 + size)
        _, initial_cost = evaluate_model(
            base_model, inputs, backend, weights, params, states, current_action,
            direct_reference, subset, device,
        )
        trained, trace = train_task_loss(
            base_model, inputs, backend, weights, params, states, current_action,
            direct_reference, subset, args.tiny_updates, args.batch_size,
            args.learning_rate, args.weight_decay, args.evaluation_interval,
            device, args.seed + size,
        )
        final_action, final_cost = evaluate_model(
            trained, inputs, backend, weights, params, states, current_action,
            direct_reference, subset, device,
        )
        value = result_metrics(initial_cost, j16_cost[subset], final_cost)
        threshold = 0.80 if size <= 32 else 0.60
        passed = value["headroom_recovery"] >= threshold
        tiny_passes.append(passed)
        tiny_results[str(size)] = {
            "indices": subset.tolist(), "metrics": value, "gate_threshold": threshold,
            "gate_passed": passed, "trace": trace,
        }
        torch.save({
            "model_state_dict": trained.state_dict(), "subset_indices": subset,
            "final_action": final_action, "trace": trace,
        }, args.output_dir / f"tiny_{size}.pt")
        print(
            f"tiny={size} recovery={value['headroom_recovery']:.3f} "
            f"J {value['initial_cost']['mean']:.3f}->{value['final_cost']['mean']:.3f} "
            f"vs J16 {value['j16_cost']['mean']:.3f} pass={passed}"
        )

    full_result = None
    if all(tiny_passes):
        _, initial_train = evaluate_model(
            base_model, inputs, backend, weights, params, states, current_action,
            direct_reference, train_pool, device,
        )
        _, initial_heldout = evaluate_model(
            base_model, inputs, backend, weights, params, states, current_action,
            direct_reference, heldout, device,
        )
        trained, trace = train_task_loss(
            base_model, inputs, backend, weights, params, states, current_action,
            direct_reference, train_pool, args.full_updates, args.batch_size,
            args.learning_rate, args.weight_decay, args.evaluation_interval,
            device, args.seed + 1000,
        )
        train_action, train_cost = evaluate_model(
            trained, inputs, backend, weights, params, states, current_action,
            direct_reference, train_pool, device,
        )
        heldout_action, heldout_cost = evaluate_model(
            trained, inputs, backend, weights, params, states, current_action,
            direct_reference, heldout, device,
        )
        full_result = {
            "train": result_metrics(initial_train, j16_cost[train_pool], train_cost),
            "episode_heldout": result_metrics(
                initial_heldout, j16_cost[heldout], heldout_cost
            ),
            "trace": trace,
        }
        torch.save({
            "model_state_dict": trained.state_dict(), "fold": args.fold,
            "seed": args.seed, "train_indices": train_pool,
            "heldout_indices": heldout, "train_action": train_action,
            "heldout_action": heldout_action, "trace": trace,
        }, args.output_dir / "full_train_fold0.pt")
        qualification = (
            "DBM_TASK_LOSS_ACTOR_FULL_TRAIN_CAPACITY_CONFIRMED"
            if full_result["train"]["headroom_recovery"] >= 0.70
            else "DBM_TASK_LOSS_TINY_PASS_FULL_TRAIN_CONDITIONING_LIMIT"
        )
    else:
        qualification = "DBM_TASK_LOSS_ACTOR_TINY_CAPACITY_GATE_FAIL"

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "contract": {
            "split": "train-only episode-grouped fold-0",
            "loss": "mean deterministic DBM J50 (positive scalar normalized)",
            "critic_used": False, "teacher_used_for_training": False,
            "j16_used_only_for_evaluation": True,
            "formal_validation_loaded": False, "test_loaded": False,
        },
        "sources": {
            "actor_checkpoint": str(checkpoint_path.resolve()),
            "actor_checkpoint_sha256": sha256_file(checkpoint_path),
            "candidate_bank": str((args.bank_root / "candidate_bank.npz").resolve()),
            "candidate_bank_sha256": sha256_file(args.bank_root / "candidate_bank.npz"),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_support_audit": {
            "component_outside_fraction": float(np.mean(outside)),
            "state_any_component_outside_fraction": float(np.mean(outside.reshape(len(outside), -1).any(1))),
            "projected_j16_cost": distribution(projected_cost),
            "raw_j16_cost": distribution(j16_cost),
            "projection_excess_cost": distribution(projected_cost - j16_cost),
        },
        "tiny": tiny_results,
        "full": full_result,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(qualification)


if __name__ == "__main__":
    main()
