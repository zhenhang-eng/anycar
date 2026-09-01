#!/usr/bin/env python3
"""OAC-0/OAC-1 pilot for the absolute-action online Actor--Critic route.

This entry point intentionally stops before any Actor update.  It freezes the
clean/no-anchor G-X Actor, samples actor-visited absolute 8x2 knot plans, obtains
their deterministic DBM J_direct values, appends every good and bad action to a
replay, and continually updates two scalar cost Critics plus an independent
flat/stay classifier.

The problem is a contextual bandit: the fixed state is not advanced and there
is no Bellman bootstrap or target network.  Formal validation/test data are not
loaded.  A later OAC-2 entry point may update the Actor only if this script's
four burn-in gates pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
)
from train_mppi_direct_trust_region_actor import load_actor_payload
from mppi_a2_actors import DirectNoAnchorGTXActor


DEFAULT_BANK = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"
)
DEFAULT_ACTOR = Path("outputs/mppi_proposal/j16_noanchor_gt_x_20260820_v1")
DEFAULT_BASE_AC = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_GT_V1 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac01_20260820_v1"
)
BASE_SIGMA = np.asarray((0.25, 0.35), np.float32)
ROLE_NAMES = (
    "actor_mean", "antithetic_0_minus", "antithetic_0_plus",
    "antithetic_1_minus", "antithetic_1_plus", "wide_exploration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--base-ac", type=Path, default=DEFAULT_BASE_AC)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--contexts-per-round", type=int, default=256)
    parser.add_argument("--critic-updates-per-round", type=int, default=20)
    parser.add_argument("--flat-pretrain-updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flat-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--material-gap", type=float, default=0.1)
    parser.add_argument("--flat-gap", type=float, default=0.1)
    parser.add_argument("--exploration-scale", type=float, default=0.25)
    parser.add_argument("--wide-exploration-scale", type=float, default=0.50)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def json_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def distribution(x: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(x, np.float64)
    return {
        "count": int(len(value)), "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)), "maximum": float(np.max(value)),
    }


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def load_bank(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    if data["actions"].shape != (1800, 24, 8, 2):
        raise AssertionError(f"unexpected candidate bank {data['actions'].shape}")
    return data


def load_actor_normalization(base_ac: Path) -> tuple[MPPIProposalNormalization, Path]:
    base = torch.load(base_ac, map_location="cpu")
    alpha = torch.load(Path(base["base_alpha_checkpoint"]), map_location="cpu")
    old_path = Path(alpha["old_actor"])
    old = load_actor_payload(old_path)
    return MPPIProposalNormalization.from_dict(old["state_normalization"]), old_path


def make_actor_inputs(
    data: dict[str, np.ndarray], normalization: MPPIProposalNormalization,
) -> tuple[np.ndarray, ...]:
    history, reference, current = normalization.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32),
        current.astype(np.float32), np.zeros((count, 8, 2), np.float32),
        np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    )


def load_frozen_actor(
    actor_root: Path, fold: int, seed: int, device: torch.device,
) -> tuple[DirectNoAnchorGTXActor, Path]:
    path = actor_root / f"a0_fold{fold}_seed{seed}.pt"
    payload = torch.load(path, map_location=device)
    if int(payload["fold"]) != fold or int(payload["seed"]) != seed:
        raise AssertionError(f"actor fold/seed mismatch: {path}")
    actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    actor.load_state_dict(payload["model_state_dict"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, path


def actor_mean(
    actor: DirectNoAnchorGTXActor, inputs: tuple[np.ndarray, ...], indices: np.ndarray,
    device: torch.device, batch_size: int = 256,
) -> np.ndarray:
    values = []
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            local = indices[begin:begin + batch_size]
            tensors = tuple(
                torch.from_numpy(value[local]).to(device) for value in inputs
            )
            _, center = actor(*tensors)
            values.append(center.cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def stratified_contexts(
    data: dict[str, np.ndarray], train: np.ndarray, count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cells: dict[tuple[float, str], np.ndarray] = {}
    for speed in sorted(np.unique(data["speed"][train])):
        for scenario in sorted(np.unique(data["scenario"][train])):
            index = train[
                (data["speed"][train] == speed)
                & (data["scenario"][train] == scenario)
            ]
            if len(index):
                cells[(float(speed), str(scenario))] = index
    keys = sorted(cells)
    selected = []
    offset = int(rng.integers(len(keys)))
    for item in range(count):
        key = keys[(item + offset) % len(keys)]
        selected.append(int(rng.choice(cells[key])))
    rng.shuffle(selected)
    return np.asarray(selected, np.int64)


def explore_actions(
    mean: np.ndarray, rng: np.random.Generator, scale: float, wide: float,
) -> np.ndarray:
    count = len(mean)
    local_std = BASE_SIGMA.reshape(1, 1, 2) * float(scale)
    wide_std = BASE_SIGMA.reshape(1, 1, 2) * float(wide)
    eps0 = rng.normal(size=(count, 8, 2)).astype(np.float32) * local_std
    eps1 = rng.normal(size=(count, 8, 2)).astype(np.float32) * local_std
    broad = rng.normal(size=(count, 8, 2)).astype(np.float32) * wide_std
    return np.stack((
        mean,
        np.clip(mean - eps0, -1.0, 1.0),
        np.clip(mean + eps0, -1.0, 1.0),
        np.clip(mean - eps1, -1.0, 1.0),
        np.clip(mean + eps1, -1.0, 1.0),
        np.clip(mean + broad, -1.0, 1.0),
    ), axis=1).astype(np.float32)


def rollout_bank(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    actions: np.ndarray,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result = []
    for begin in range(0, len(indices), batch_size):
        local = indices[begin:begin + batch_size]
        knots = torch.from_numpy(actions[begin:begin + batch_size]).to(device)
        full_actions = interpolate_knots(knots, params.horizon)
        with torch.no_grad():
            value = batched_cost(
                backend, weights, full_actions,
                torch.from_numpy(states[local]).to(device),
                torch.from_numpy(current[local]).to(device),
                torch.from_numpy(references[local]).to(device),
            )
        result.append(value.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


class ActorVisitedReplay:
    def __init__(self) -> None:
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.cost: list[np.ndarray] = []
        self.round: list[np.ndarray] = []
        self.group: list[np.ndarray] = []
        self.role: list[np.ndarray] = []
        self.pre1: list[np.ndarray] = []
        self.pre2: list[np.ndarray] = []

    def add(
        self, state: np.ndarray, action: np.ndarray, cost: np.ndarray,
        round_index: int, group: np.ndarray, role: np.ndarray,
        pre1: np.ndarray, pre2: np.ndarray,
    ) -> None:
        size = len(state)
        self.state.append(np.asarray(state, np.int64))
        self.action.append(np.asarray(action, np.float32))
        self.cost.append(np.asarray(cost, np.float32))
        self.round.append(np.full(size, round_index, np.int16))
        self.group.append(np.asarray(group, np.int32))
        self.role.append(np.asarray(role))
        self.pre1.append(np.asarray(pre1, np.float32))
        self.pre2.append(np.asarray(pre2, np.float32))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "state_index": np.concatenate(self.state),
            "action": np.concatenate(self.action),
            "cost": np.concatenate(self.cost),
            "round": np.concatenate(self.round),
            "interaction_group": np.concatenate(self.group),
            "role": np.concatenate(self.role),
            "pre_critic1": np.concatenate(self.pre1),
            "pre_critic2": np.concatenate(self.pre2),
        }


def critic_state_inputs(
    data: dict[str, np.ndarray], payload: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(
        payload["training"]["normalization"]
    )
    values = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    return tuple(np.asarray(value, np.float32) for value in values)


def load_critic(
    root: Path, seed: int, fold: int, device: torch.device,
) -> tuple[AbsoluteActionValueCritic, dict[str, Any], Path]:
    path = root / f"critic_seed{seed}_fold{fold}.pt"
    payload = torch.load(path, map_location=device)
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload, path


def predict_actions(
    model: AbsoluteActionValueCritic,
    inputs: tuple[np.ndarray, ...],
    payload: dict[str, Any],
    state: np.ndarray,
    action: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(state), batch_size):
            local = slice(begin, begin + batch_size)
            index = state[local]
            prediction = model(
                torch.from_numpy(inputs[0][index]).to(device),
                torch.from_numpy(inputs[1][index]).to(device),
                torch.from_numpy(inputs[2][index]).to(device),
                torch.from_numpy(action[local, None]).to(device),
            )[:, 0]
            prediction = (
                prediction * float(payload["training"]["target_std"])
                + float(payload["training"]["target_mean"])
            )
            result.append(prediction.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def predict_flat(
    model: AbsoluteActionValueCritic,
    inputs: tuple[np.ndarray, ...],
    state: np.ndarray,
    action: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(state), batch_size):
            local = slice(begin, begin + batch_size)
            index = state[local]
            logit = model(
                torch.from_numpy(inputs[0][index]).to(device),
                torch.from_numpy(inputs[1][index]).to(device),
                torch.from_numpy(inputs[2][index]).to(device),
                torch.from_numpy(action[local, None]).to(device),
            )[:, 0]
            result.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def material_pair_accuracy(
    prediction: np.ndarray, cost: np.ndarray, group: np.ndarray, gap: float,
) -> tuple[float, int]:
    correct: list[bool] = []
    for value in np.unique(group):
        index = np.flatnonzero(group == value)
        if len(index) < 2:
            continue
        i, j = np.triu_indices(len(index), 1)
        left, right = index[i], index[j]
        mask = np.abs(cost[left] - cost[right]) >= gap
        correct.extend((
            np.sign(prediction[left[mask]] - prediction[right[mask]])
            == np.sign(cost[left[mask]] - cost[right[mask]])
        ).tolist())
    return (float(np.mean(correct)) if correct else 0.0, len(correct))


def sample_training_points(
    data: dict[str, np.ndarray], train: np.ndarray, replay: dict[str, np.ndarray],
    round_index: int, batch_size: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recent_count = batch_size // 2
    historical_count = batch_size // 4
    hard_count = batch_size - recent_count - historical_count
    recent_pool = np.flatnonzero(replay["round"] >= max(0, round_index - 1))
    recent = rng.choice(recent_pool, recent_count, replace=len(recent_pool) < recent_count)

    historical_state = rng.choice(train, historical_count, replace=True)
    historical_candidate = rng.integers(24, size=historical_count)
    h_action = data["actions"][historical_state, historical_candidate]
    h_cost = data["costs"][historical_state, historical_candidate]

    priority = replay["cost"] + 2.0 * np.abs(replay["pre_critic1"] - replay["pre_critic2"])
    threshold = np.quantile(priority, 0.75)
    hard_pool = np.flatnonzero(priority >= threshold)
    hard = rng.choice(hard_pool, hard_count, replace=len(hard_pool) < hard_count)

    return (
        np.concatenate((replay["state_index"][recent], historical_state,
                        replay["state_index"][hard])),
        np.concatenate((replay["action"][recent], h_action,
                        replay["action"][hard])),
        np.concatenate((replay["cost"][recent], h_cost,
                        replay["cost"][hard])),
    )


def sample_pairs(
    data: dict[str, np.ndarray], train: np.ndarray, replay: dict[str, np.ndarray],
    pair_count: int, gap: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, costs = [], [], []
    group_key = replay["interaction_group"]
    groups = np.unique(group_key)
    attempts = 0
    while len(states) < pair_count // 2 and attempts < pair_count * 20:
        attempts += 1
        key = rng.choice(groups)
        candidates = np.flatnonzero(group_key == key)
        pair = rng.choice(candidates, 2, replace=False)
        if abs(float(replay["cost"][pair[0]] - replay["cost"][pair[1]])) < gap:
            continue
        states.append(int(replay["state_index"][pair[0]]))
        actions.append(replay["action"][pair])
        costs.append(replay["cost"][pair])
    while len(states) < pair_count:
        state = int(rng.choice(train))
        pair = rng.choice(24, 2, replace=False)
        if abs(float(data["costs"][state, pair[0]] - data["costs"][state, pair[1]])) < gap:
            continue
        states.append(state)
        actions.append(data["actions"][state, pair])
        costs.append(data["costs"][state, pair])
    return (
        np.asarray(states, np.int64), np.asarray(actions, np.float32),
        np.asarray(costs, np.float32),
    )


def update_value_critic(
    model: AbsoluteActionValueCritic,
    optimizer: torch.optim.Optimizer,
    inputs: tuple[np.ndarray, ...],
    payload: dict[str, Any],
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
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
    value_loss = F.smooth_l1_loss(prediction, target)
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
    loss = value_loss + args.ranking_weight * ranking
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()), "value": float(value_loss.detach()),
        "ranking": float(ranking.detach()),
    }


def sample_flat_batch(
    data: dict[str, np.ndarray], train: np.ndarray,
    replay: dict[str, np.ndarray] | None, batch_size: int, flat_gap: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    historical_count = batch_size if replay is None else batch_size // 2
    states = rng.choice(train, historical_count, replace=True)
    candidates = rng.integers(24, size=historical_count)
    action = data["actions"][states, candidates]
    label = data["costs"][states, candidates] <= (
        data["costs"][states].min(axis=1) + flat_gap
    )
    if replay is not None:
        online_count = batch_size - historical_count
        chosen = rng.choice(len(replay["cost"]), online_count, replace=True)
        online_label = np.zeros(online_count, bool)
        group = replay["interaction_group"]
        for position, row in enumerate(chosen):
            members = np.flatnonzero(group == group[row])
            online_label[position] = replay["cost"][row] <= (
                replay["cost"][members].min() + flat_gap
            )
        states = np.concatenate((states, replay["state_index"][chosen]))
        action = np.concatenate((action, replay["action"][chosen]))
        label = np.concatenate((label, online_label))
    return states, action.astype(np.float32), label.astype(np.float32)


def update_flat_head(
    model: AbsoluteActionValueCritic,
    optimizer: torch.optim.Optimizer,
    inputs: tuple[np.ndarray, ...],
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> float:
    model.train()
    state, action, label = batch
    logit = model(
        torch.from_numpy(inputs[0][state]).to(device),
        torch.from_numpy(inputs[1][state]).to(device),
        torch.from_numpy(inputs[2][state]).to(device),
        torch.from_numpy(action[:, None]).to(device),
    )[:, 0]
    target = torch.from_numpy(label).to(device)
    positives = max(float(target.sum()), 1.0)
    negatives = max(float(len(target) - target.sum()), 1.0)
    positive_weight = torch.tensor(
        min(negatives / positives, 10.0), device=device
    )
    loss = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=positive_weight
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())


def final_metrics(
    data: dict[str, np.ndarray], replay: dict[str, np.ndarray],
    pred1: np.ndarray, pred2: np.ndarray, flat: np.ndarray,
    flat_bank_best: np.ndarray, flat_warm: np.ndarray, material_gap: float,
    flat_gap: float,
) -> dict[str, Any]:
    conservative = np.maximum(pred1, pred2)
    group = replay["interaction_group"]
    pair_accuracy, pair_count = material_pair_accuracy(
        conservative, replay["cost"], group, material_gap
    )
    target = np.log1p(replay["cost"])
    std_ratio = [
        float(np.std(prediction) / max(np.std(target), 1e-12))
        for prediction in (pred1, pred2)
    ]
    warm_gap = data["costs"][:, 0] - data["costs"].min(axis=1)
    warm_material = warm_gap > flat_gap
    flat_metrics = {
        "bank_best_recall": float(np.mean(flat_bank_best >= 0.5)),
        "warm_false_stay": float(np.mean(flat_warm[warm_material] >= 0.5)),
        "warm_material_count": int(warm_material.sum()),
    }

    pre = np.maximum(replay["pre_critic1"], replay["pre_critic2"])
    eligible = replay["round"] <= int(replay["round"].max()) - 2
    bad_rows, mean_rows = [], []
    for key in np.unique(group[eligible]):
        members = np.flatnonzero(group == key)
        mean = members[replay["role"][members] == "actor_mean"]
        if len(mean) != 1:
            raise AssertionError("each interaction group must have one actor mean")
        bad = members[replay["cost"][members] > replay["cost"][mean[0]] + material_gap]
        bad_rows.extend(bad.tolist())
        mean_rows.extend([int(mean[0])] * len(bad))
    bad_rows = np.asarray(bad_rows, np.int64)
    mean_rows = np.asarray(mean_rows, np.int64)
    if len(bad_rows):
        pre_correct = pre[bad_rows] > pre[mean_rows]
        final_correct = conservative[bad_rows] > conservative[mean_rows]
        initially_wrong = ~pre_correct
        corrected = (
            float(np.mean(final_correct[initially_wrong]))
            if initially_wrong.any() else 1.0
        )
        bad_correction = {
            "eligible_pairs": int(len(bad_rows)),
            "pre_accuracy": float(np.mean(pre_correct)),
            "lag2_final_accuracy": float(np.mean(final_correct)),
            "initially_wrong_count": int(initially_wrong.sum()),
            "initially_wrong_corrected_fraction": corrected,
        }
    else:
        bad_correction = {
            "eligible_pairs": 0, "pre_accuracy": 0.0,
            "lag2_final_accuracy": 0.0, "initially_wrong_count": 0,
            "initially_wrong_corrected_fraction": 0.0,
        }
    result = {
        "actor_visited": {
            "count": int(len(target)),
            "cost": distribution(replay["cost"]),
            "pearson": [correlation(pred1, target), correlation(pred2, target)],
            "prediction_std_ratio": std_ratio,
            "material_pair_accuracy_conservative": pair_accuracy,
            "material_pair_count": pair_count,
            "twin_abs_difference": distribution(np.abs(pred1 - pred2)),
        },
        "flat_stay": flat_metrics,
        "bad_action_correction": bad_correction,
    }
    gates = {
        "actor_visited_material_pair_accuracy_ge_0_85": pair_accuracy >= 0.85,
        "both_critics_no_value_collapse": bool(
            all(np.isfinite(result["actor_visited"]["pearson"]))
            and all(0.5 <= ratio <= 2.0 for ratio in std_ratio)
        ),
        "flat_bank_best_recall_ge_0_80": flat_metrics["bank_best_recall"] >= 0.80,
        "flat_warm_false_stay_le_0_10": flat_metrics["warm_false_stay"] <= 0.10,
        "lag2_bad_action_accuracy_ge_0_85": (
            bad_correction["lag2_final_accuracy"] >= 0.85
        ),
        "initially_wrong_bad_actions_corrected_ge_0_50": (
            bad_correction["initially_wrong_corrected_fraction"] >= 0.50
        ),
    }
    result["gates"] = gates
    result["passed"] = bool(all(gates.values()))
    return result


def write_oac0_contract(
    args: argparse.Namespace, data: dict[str, np.ndarray], folds: np.ndarray,
    params_json: str, weights_json: str, dbm_json: str, old_actor: Path,
) -> dict[str, Any]:
    seeds = [int(value) for value in args.seeds.split(",")]
    train_episodes = sorted(np.unique(data["episode"][folds != args.fold]).tolist())
    heldout_episodes = sorted(np.unique(data["episode"][folds == args.fold]).tolist())
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC0_CONTRACT_FROZEN",
        "stage": "OAC-0/OAC-1 only; Actor updates forbidden",
        "arguments": serialize_args(args),
        "fold": args.fold,
        "train_state_count": int(np.sum(folds != args.fold)),
        "heldout_state_count": int(np.sum(folds == args.fold)),
        "train_episodes": train_episodes,
        "heldout_episodes": heldout_episodes,
        "seeds": seeds,
        "actor_contract": (
            "DirectNoAnchorGTXActor; history+ego-reference+current; "
            "anchor/feedback/gradient zero; full absolute 8x2 [acceleration,steering]"
        ),
        "critic_contract": (
            "Twin AbsoluteActionValueCritic predicting log1p(J_direct); "
            "independent AbsoluteActionValueCritic flat/stay classifier"
        ),
        "interaction_contract": (
            "fixed state contextual bandit; deterministic DBM immediate cost; "
            "no next_state, target Critic, Bellman bootstrap, or Actor optimizer"
        ),
        "replay_mix": {
            "recent_actor_visited": 0.50,
            "historical_bank": 0.25,
            "hard_priority": 0.25,
        },
        "source_hashes": {
            "candidate_bank": sha256_file(args.bank_root / "candidate_bank.npz"),
            "candidate_manifest": sha256_file(args.bank_root / "dataset_manifest.json"),
            "old_actor_normalization": sha256_file(old_actor),
            "gt_v1_summary": sha256_file(args.gt_v1 / "summary.json"),
            "mppi_params_json": json_digest(params_json),
            "cost_weights_json": json_digest(weights_json),
            "dbm_params_json": json_digest(dbm_json),
        },
        "actor_checkpoints": {
            str(seed): {
                "path": str((args.actor_root / f"a0_fold{args.fold}_seed{seed}.pt").resolve()),
                "sha256": sha256_file(args.actor_root / f"a0_fold{args.fold}_seed{seed}.pt"),
            } for seed in seeds
        },
        "critic_initialization": {
            str(seed): [
                {
                    "source_seed": source_seed,
                    "path": str((args.bank_root / f"critic_seed{source_seed}_fold{args.fold}.pt").resolve()),
                    "sha256": sha256_file(
                        args.bank_root / f"critic_seed{source_seed}_fold{args.fold}.pt"
                    ),
                }
                for source_seed in (seed, (seed + 1) % 3)
            ] for seed in seeds
        },
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return contract


def run_seed(
    args: argparse.Namespace, seed: int, data: dict[str, np.ndarray],
    folds: np.ndarray, actor_inputs: tuple[np.ndarray, ...],
    states: np.ndarray, current_action: np.ndarray, references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(26082000 + seed)
    train = np.flatnonzero(folds != args.fold)
    heldout = np.flatnonzero(folds == args.fold)
    actor, actor_path = load_frozen_actor(args.actor_root, args.fold, seed, device)
    actor_hash_before = module_digest(actor)
    critic1, payload1, critic_path1 = load_critic(
        args.bank_root, seed, args.fold, device
    )
    second_seed = (seed + 1) % 3
    critic2, payload2, critic_path2 = load_critic(
        args.bank_root, second_seed, args.fold, device
    )
    inputs1 = critic_state_inputs(data, payload1)
    inputs2 = critic_state_inputs(data, payload2)
    if max(np.max(np.abs(a - b)) for a, b in zip(inputs1, inputs2)) > 1e-6:
        raise AssertionError("Twin Critic state normalizations differ for one fold")
    flat_head = AbsoluteActionValueCritic(dropout=0.0).to(device)
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
    flat_pretrain_loss = []
    for _ in range(args.flat_pretrain_updates):
        batch = sample_flat_batch(
            data, train, None, args.batch_size, args.flat_gap, rng
        )
        flat_pretrain_loss.append(update_flat_head(
            flat_head, flat_optimizer, inputs1, batch, device
        ))

    replay = ActorVisitedReplay()
    round_records = []
    for round_index in range(args.rounds):
        selected = stratified_contexts(
            data, train, args.contexts_per_round, rng
        )
        mean = actor_mean(actor, actor_inputs, selected, device)
        action_bank = explore_actions(
            mean, rng, args.exploration_scale, args.wide_exploration_scale
        )
        costs = rollout_bank(
            backend, weights, params, action_bank, states, current_action,
            references, selected, args.rollout_batch_size, device,
        )
        flat_state = np.repeat(selected, len(ROLE_NAMES))
        flat_action = action_bank.reshape(-1, 8, 2)
        flat_cost = costs.reshape(-1)
        roles = np.tile(np.asarray(ROLE_NAMES), len(selected))
        interaction_group = np.repeat(
            round_index * args.contexts_per_round + np.arange(len(selected)),
            len(ROLE_NAMES),
        ).astype(np.int32)
        pre1 = predict_actions(
            critic1, inputs1, payload1, flat_state, flat_action, device
        )
        pre2 = predict_actions(
            critic2, inputs2, payload2, flat_state, flat_action, device
        )
        replay.add(
            flat_state, flat_action, flat_cost, round_index,
            interaction_group, roles, pre1, pre2
        )
        arrays = replay.arrays()
        losses1, losses2, flat_losses = [], [], []
        for _ in range(args.critic_updates_per_round):
            points = sample_training_points(
                data, train, arrays, round_index, args.batch_size, rng
            )
            pairs = sample_pairs(
                data, train, arrays, args.pair_batch_size,
                args.material_gap, rng,
            )
            losses1.append(update_value_critic(
                critic1, optimizer1, inputs1, payload1, points, pairs, args, device
            ))
            losses2.append(update_value_critic(
                critic2, optimizer2, inputs2, payload2, points, pairs, args, device
            ))
            flat_losses.append(update_flat_head(
                flat_head, flat_optimizer, inputs1,
                sample_flat_batch(
                    data, train, arrays, args.batch_size, args.flat_gap, rng
                ), device,
            ))
        post1 = predict_actions(
            critic1, inputs1, payload1, flat_state, flat_action, device
        )
        post2 = predict_actions(
            critic2, inputs2, payload2, flat_state, flat_action, device
        )
        group = np.repeat(np.arange(len(selected)), len(ROLE_NAMES))
        accuracy, pair_count = material_pair_accuracy(
            np.maximum(post1, post2), flat_cost, group, args.material_gap
        )
        record = {
            "round": round_index,
            "new_rows": int(len(flat_cost)),
            "cost": distribution(flat_cost),
            "material_pair_accuracy": accuracy,
            "material_pair_count": pair_count,
            "critic1_loss": float(np.mean([value["loss"] for value in losses1])),
            "critic2_loss": float(np.mean([value["loss"] for value in losses2])),
            "flat_loss": float(np.mean(flat_losses)),
        }
        round_records.append(record)
        print(
            f"seed={seed} round={round_index + 1}/{args.rounds} "
            f"rows={len(arrays['cost'])} pair={accuracy:.3f} "
            f"Jmean={np.mean(flat_cost):.3f}", flush=True,
        )

    arrays = replay.arrays()
    pred1 = predict_actions(
        critic1, inputs1, payload1, arrays["state_index"], arrays["action"], device
    )
    pred2 = predict_actions(
        critic2, inputs2, payload2, arrays["state_index"], arrays["action"], device
    )
    flat_probability = predict_flat(
        flat_head, inputs1, arrays["state_index"], arrays["action"], device
    )
    best_index = np.argmin(data["costs"], axis=1)
    bank_best_action = data["actions"][np.arange(len(data["costs"])), best_index]
    flat_bank_best = predict_flat(
        flat_head, inputs1, np.arange(len(data["costs"])), bank_best_action, device
    )
    flat_warm = predict_flat(
        flat_head, inputs1, np.arange(len(data["costs"])), data["actions"][:, 0], device
    )
    metrics = final_metrics(
        data, arrays, pred1, pred2, flat_probability,
        flat_bank_best, flat_warm, args.material_gap, args.flat_gap,
    )
    actor_hash_after = module_digest(actor)
    if actor_hash_after != actor_hash_before:
        raise AssertionError("frozen Actor changed during OAC-1")
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir()
    arrays.update({
        "final_critic1": pred1, "final_critic2": pred2,
        "flat_probability": flat_probability,
    })
    np.savez_compressed(seed_dir / "actor_visited_replay.npz", **arrays)
    checkpoint_common = {
        "fold": args.fold, "pilot_seed": seed,
        "actor_update_count": 0,
        "actor_sha256_before": actor_hash_before,
        "actor_sha256_after": actor_hash_after,
        "formal_validation_loaded": False, "test_loaded": False,
    }
    torch.save({
        **checkpoint_common, "model_class": "AbsoluteActionValueCritic",
        "model": critic1.state_dict(), "training": payload1["training"],
        "source_checkpoint": str(critic_path1.resolve()),
    }, seed_dir / "critic1.pt")
    torch.save({
        **checkpoint_common, "model_class": "AbsoluteActionValueCritic",
        "model": critic2.state_dict(), "training": payload2["training"],
        "source_checkpoint": str(critic_path2.resolve()),
    }, seed_dir / "critic2.pt")
    torch.save({
        **checkpoint_common, "model_class": "AbsoluteActionValueCriticFlatHead",
        "model": flat_head.state_dict(),
        "normalization": payload1["training"]["normalization"],
    }, seed_dir / "flat_head.pt")
    result = {
        "seed": seed, "fold": args.fold,
        "qualification": "OAC1_BURNIN_PASS" if metrics["passed"] else "OAC1_BURNIN_FAIL",
        "actor": {
            "checkpoint": str(actor_path.resolve()),
            "checkpoint_sha256": sha256_file(actor_path),
            "module_sha256_before": actor_hash_before,
            "module_sha256_after": actor_hash_after,
            "update_count": 0,
        },
        "initial_critics": [str(critic_path1.resolve()), str(critic_path2.resolve())],
        "replay": {
            "rows": int(len(arrays["cost"])),
            "role_counts": {
                str(key): int(value)
                for key, value in Counter(arrays["role"].tolist()).items()
            },
            "train_only": bool(np.all(folds[arrays["state_index"]] != args.fold)),
        },
        "flat_pretrain_final_loss": float(np.mean(flat_pretrain_loss[-20:])),
        "rounds": round_records,
        "metrics": metrics,
    }
    (seed_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fold not in (0, 1, 2):
        raise ValueError("outer fold must be one of 0, 1, or 2")
    if args.rounds != 10 or args.critic_updates_per_round != 20:
        raise ValueError("OAC-1 contract requires exactly 10 rounds and 20 Critic updates/round")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    states, current_action, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(dbm_json))
    )
    normalization, old_actor = load_actor_normalization(args.base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    contract = write_oac0_contract(
        args, data, folds, params_json, weights_json, dbm_json, old_actor
    )
    records = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        records.append(run_seed(
            args, seed, data, folds, actor_inputs, states, current_action,
            references, backend, weights, params, device,
        ))
    passed = sum(row["qualification"] == "OAC1_BURNIN_PASS" for row in records)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC1_BURNIN_GATE_PASS_ACTOR_MAY_ENTER_OAC2"
            if passed >= 2 else "OAC1_BURNIN_GATE_FAIL_ACTOR_REMAINS_FROZEN"
        ),
        "contract": str((args.output_dir / "contract.json").resolve()),
        "contract_qualification": contract["qualification"],
        "passed_seed_count": passed,
        "required_seed_count": 2,
        "actor_update_count": 0,
        "records": records,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": result["qualification"],
        "passed_seed_count": passed,
        "seed_qualifications": {
            str(row["seed"]): row["qualification"] for row in records
        },
    }, indent=2))


if __name__ == "__main__":
    main()
