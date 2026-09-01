#!/usr/bin/env python3
"""Frozen-Actor Critic capacity and pairwise-delta A/B on fixed OAC replay.

The experiment compares three arms without constructing an Actor optimizer:

``base``
    The current absolute-action scalar value architecture and loss.
``wide``
    A roughly two-parameter-count capacity arm with the identical loss.
``pair_delta``
    The base absolute value model plus an antisymmetric same-state pairwise
    ``Delta log(1 + J)`` head.  The absolute value/ranking losses are retained;
    the new head receives direct delta regression and ranking supervision.

All arms see the same train-only candidate bank, the same fixed long-horizon
OAC Actor-visited replay, and the same sampled mini-batches.  Evaluation uses
the outer episode-heldout bank plus initial/selected/latest Actor candidates
whose deterministic DBM costs are recomputed once and saved.  Formal
validation and test data are never loaded.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import TemporalConvEncoder
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from generate_dbm_proposal_teacher import sha256_file
from mppi_a2_actors import DirectNoAnchorGTXSupportActor
from run_mppi_absolute_action_value_critic_cv import (
    FLAT_SENSITIVITY,
    KNOT_TIMES,
    make_folds,
    metrics as bank_metrics,
)
from train_mppi_oac2_continuous_actor import load_actor
from train_mppi_online_absolute_sac import (
    actor_mean,
    critic_state_inputs,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    rollout_bank,
    sample_pairs,
    sample_training_points,
)
from validate_mppi_oac2_continuous_actor import load_actor_checkpoint


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
)
DEFAULT_BANK = Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1")
DEFAULT_ACTOR = Path("outputs/mppi_proposal/j16_noanchor_gt_x_20260820_v1")
DEFAULT_BASE_AC = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_GT_V1 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1"
)
ROLES = ("initial", "selected", "latest")
ARMS = ("base", "wide", "pair_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--base-ac", type=Path, default=DEFAULT_BASE_AC)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--updates", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--pair-delta-weight", type=float, default=1.0)
    parser.add_argument("--material-gap", type=float, default=0.1)
    parser.add_argument("--report-every", type=int, default=400)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, bool).reshape(-1)
    score = np.asarray(score, np.float64).reshape(-1)
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else 0.5


def distribution(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, np.float64).reshape(-1)
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def knot_geometry(reference: torch.Tensor) -> torch.Tensor:
    samples = reference[:, KNOT_TIMES, :]
    return torch.stack((
        samples[..., 0],
        samples[..., 1],
        torch.atan2(samples[..., 2], samples[..., 3]),
        samples[..., 4],
    ), dim=-1)


class ConfigurableAbsoluteActionValueCritic(nn.Module):
    """Current scalar-value topology with an explicit capacity configuration."""

    def __init__(self, wide: bool = False, pair_delta: bool = False) -> None:
        super().__init__()
        if wide:
            temporal_dim, current_dim = 200, 104
            state_dim, token_dim, feedforward_dim = 304, 96, 192
            value_hidden, value_tail = 304, 104
        else:
            temporal_dim, current_dim = 128, 64
            state_dim, token_dim, feedforward_dim = 192, 64, 128
            value_hidden, value_tail = 192, 64
        self.config = {
            "wide": bool(wide), "pair_delta": bool(pair_delta),
            "temporal_dim": temporal_dim, "current_dim": current_dim,
            "state_dim": state_dim, "token_dim": token_dim,
            "feedforward_dim": feedforward_dim,
            "transformer_layers": 2,
        }
        self.pair_delta_enabled = bool(pair_delta)
        self.history_encoder = TemporalConvEncoder(7, 8, temporal_dim, 0.0)
        self.reference_encoder = TemporalConvEncoder(5, 5, temporal_dim, 0.0)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, current_dim), nn.SiLU(),
            nn.Linear(current_dim, current_dim), nn.SiLU(),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(2 * temporal_dim + current_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, state_dim), nn.SiLU(),
        )
        self.state_projection = nn.Linear(state_dim, token_dim)
        self.token_projection = nn.Linear(9, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4, dim_feedforward=feedforward_dim,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.value_head = nn.Sequential(
            nn.Linear(token_dim + state_dim + 16, value_hidden), nn.SiLU(),
            nn.Linear(value_hidden, value_tail), nn.SiLU(),
            nn.Linear(value_tail, 1),
        )
        if self.pair_delta_enabled:
            # Symmetrization in pair_delta() makes this an antisymmetric
            # comparator and guarantees delta(a, a) == 0 exactly.
            pair_input = state_dim + 2 * token_dim + 16
            self.pair_head = nn.Sequential(
                nn.Linear(pair_input, value_hidden), nn.SiLU(),
                nn.Linear(value_hidden, value_tail), nn.SiLU(),
                nn.Linear(value_tail, 1),
            )
        times = torch.tensor(KNOT_TIMES, dtype=torch.float32) / 49.0
        self.register_buffer("time_encoding", torch.stack((times, 1.0 - times), -1))
        sensitivity = torch.tensor(FLAT_SENSITIVITY).reshape(8, 2).sum(1)
        self.register_buffer("sensitivity", sensitivity / sensitivity.max())

    def encode_state(self, history, reference, current):
        return self.state_fusion(torch.cat((
            self.history_encoder(history),
            self.reference_encoder(reference),
            self.current_encoder(current),
        ), dim=-1))

    def encode_actions(self, state, reference, actions):
        batch, candidates = actions.shape[:2]
        geometry = knot_geometry(reference)
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.sensitivity[None, :, None].expand(batch, -1, -1)
        static = torch.cat((
            actions,
            times[:, None].expand(-1, candidates, -1, -1),
            sensitivity[:, None].expand(-1, candidates, -1, -1),
            geometry[:, None].expand(-1, candidates, -1, -1),
        ), dim=-1)
        token = self.token_projection(static.reshape(batch * candidates, 8, 9))
        state_token = self.state_projection(state)[:, None, None, :].expand(
            -1, candidates, 8, -1
        ).reshape(batch * candidates, 8, -1)
        return self.action_encoder(token + state_token).mean(1).reshape(
            batch, candidates, -1
        )

    def values_from_encoded(self, state, encoded, actions):
        batch, candidates = actions.shape[:2]
        state_expanded = state[:, None].expand(-1, candidates, -1)
        value = self.value_head(torch.cat((
            encoded, state_expanded, actions.reshape(batch, candidates, 16)
        ), dim=-1))
        return value[..., 0]

    def forward(self, history, reference, current, actions):
        state = self.encode_state(history, reference, current)
        encoded = self.encode_actions(state, reference, actions)
        return self.values_from_encoded(state, encoded, actions)

    def _raw_pair(self, state, left_encoded, right_encoded, left, right):
        return self.pair_head(torch.cat((
            state,
            left_encoded + right_encoded,
            left_encoded - right_encoded,
            (left - right).reshape(len(left), 16),
        ), dim=-1))[:, 0]

    def pair_delta(self, history, reference, current, left, right):
        if not self.pair_delta_enabled:
            raise RuntimeError("pair_delta head is not enabled")
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError("left/right must both be [B,8,2]")
        state = self.encode_state(history, reference, current)
        actions = torch.stack((left, right), dim=1)
        encoded = self.encode_actions(state, reference, actions)
        forward = self._raw_pair(
            state, encoded[:, 0], encoded[:, 1], left, right
        )
        reverse = self._raw_pair(
            state, encoded[:, 1], encoded[:, 0], right, left
        )
        return 0.5 * (forward - reverse)


def model_for_arm(arm: str) -> ConfigurableAbsoluteActionValueCritic:
    if arm == "base":
        return ConfigurableAbsoluteActionValueCritic()
    if arm == "wide":
        return ConfigurableAbsoluteActionValueCritic(wide=True)
    if arm == "pair_delta":
        return ConfigurableAbsoluteActionValueCritic(pair_delta=True)
    raise ValueError(arm)


def tensor_inputs(inputs, index, device):
    return tuple(torch.from_numpy(value[index]).to(device) for value in inputs)


def update_model(
    model, optimizer, inputs, target_mean, target_std, points, pairs,
    args, device,
) -> dict[str, float]:
    model.train()
    state, action, cost = points
    pair_state, pair_action, pair_cost = pairs
    h, r, c = tensor_inputs(inputs, state, device)
    prediction = model(
        h, r, c, torch.from_numpy(action[:, None]).to(device)
    )[:, 0]
    target = (
        torch.log1p(torch.from_numpy(cost).to(device)) - target_mean
    ) / target_std
    value_loss = F.smooth_l1_loss(prediction, target)

    ph, pr, pc = tensor_inputs(inputs, pair_state, device)
    pair_action_t = torch.from_numpy(pair_action).to(device)
    pair_cost_t = torch.from_numpy(pair_cost).to(device)
    pair_prediction = model(ph, pr, pc, pair_action_t)
    true_delta_raw = pair_cost_t[:, 0] - pair_cost_t[:, 1]
    absolute_delta = pair_prediction[:, 0] - pair_prediction[:, 1]
    rank_loss = F.softplus(
        -true_delta_raw.sign() * absolute_delta / args.ranking_temperature
    ).mean()
    pair_loss = prediction.new_zeros(())
    pair_rank = prediction.new_zeros(())
    if model.pair_delta_enabled:
        predicted_delta = model.pair_delta(
            ph, pr, pc, pair_action_t[:, 0], pair_action_t[:, 1]
        )
        true_delta = (
            torch.log1p(pair_cost_t[:, 0]) - torch.log1p(pair_cost_t[:, 1])
        ) / target_std
        pair_loss = F.smooth_l1_loss(predicted_delta, true_delta)
        pair_rank = F.softplus(
            -true_delta_raw.sign() * predicted_delta / args.ranking_temperature
        ).mean()
    loss = (
        value_loss + args.ranking_weight * rank_loss
        + args.pair_delta_weight * pair_loss
        + args.ranking_weight * pair_rank
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "value": float(value_loss.detach()),
        "absolute_ranking": float(rank_loss.detach()),
        "pair_delta": float(pair_loss.detach()),
        "pair_ranking": float(pair_rank.detach()),
    }


def predict_absolute(model, inputs, states, actions, target_mean, target_std,
                     device, batch_size=128):
    result = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(states), batch_size):
            local = slice(begin, begin + batch_size)
            h, r, c = tensor_inputs(inputs, states[local], device)
            value = model(
                h, r, c, torch.from_numpy(actions[local]).to(device)
            )
            result.append((value * target_std + target_mean).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def predict_pair_scores(model, inputs, states, actions, target_std, device,
                        batch_size=128):
    if not model.pair_delta_enabled:
        raise RuntimeError("pair scores require pair_delta arm")
    count, candidates = actions.shape[:2]
    scores = np.zeros((count, candidates), np.float32)
    model.eval()
    with torch.no_grad():
        for candidate in range(1, candidates):
            values = []
            for begin in range(0, count, batch_size):
                local = slice(begin, begin + batch_size)
                h, r, c = tensor_inputs(inputs, states[local], device)
                delta = model.pair_delta(
                    h, r, c,
                    torch.from_numpy(actions[local, candidate]).to(device),
                    torch.from_numpy(actions[local, 0]).to(device),
                )
                values.append((delta * target_std).cpu().numpy())
            scores[:, candidate] = np.concatenate(values)
    return scores


def predict_pair_delta(model, inputs, states, left, right, target_std, device,
                       batch_size=128):
    """Predict physical log-value delta ``left - right`` directly."""
    values = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(states), batch_size):
            local = slice(begin, begin + batch_size)
            h, r, c = tensor_inputs(inputs, states[local], device)
            delta = model.pair_delta(
                h, r, c,
                torch.from_numpy(left[local]).to(device),
                torch.from_numpy(right[local]).to(device),
            )
            values.append((delta * target_std).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def delta_metrics(score, costs):
    true_delta = np.log1p(costs[:, 1]) - np.log1p(costs[:, 0])
    raw_delta = costs[:, 1] - costs[:, 0]
    material = np.abs(raw_delta) >= 0.1
    regress = raw_delta > 0
    severe = raw_delta >= 10.0
    predicted_regress = score > 0
    return {
        "pearson_log_delta": correlation(score, true_delta),
        "sign_accuracy": float(np.mean(np.sign(score) == np.sign(true_delta))),
        "material_sign_accuracy": float(np.mean(
            np.sign(score[material]) == np.sign(true_delta[material])
        )) if np.any(material) else 0.0,
        "regress_auc": safe_auc(regress, score),
        "regress_recall_at_zero": float(np.mean(predicted_regress[regress]))
        if np.any(regress) else 1.0,
        "severe_count": int(severe.sum()),
        "severe_recall_at_zero": float(np.mean(predicted_regress[severe]))
        if np.any(severe) else 1.0,
        "true_delta": distribution(true_delta),
        "predicted_delta": distribution(score),
    }


def candidate_selection_metrics(score, costs):
    selected = np.argmin(score, axis=1)
    row = np.arange(len(costs))
    chosen = costs[row, selected]
    warm = costs[:, 0]
    oracle = costs.min(axis=1)
    denominator = float(np.sum(warm - oracle))
    return {
        "top1_cost_mean": float(np.mean(chosen)),
        "top1_regret_mean": float(np.mean(chosen - oracle)),
        "top1_headroom_recovery": float(np.sum(warm - chosen) / denominator)
        if abs(denominator) > 1e-12 else 0.0,
        "harmful_top1_fraction": float(np.mean(chosen > warm + 1e-5)),
        "near_best_top1_fraction": float(np.mean(chosen <= oracle * 1.005 + 1e-5)),
    }


def evaluate_arm(model, inputs, states, bank_actions, bank_costs,
                 actor_actions, actor_costs, speed, target_mean, target_std,
                 device):
    absolute_bank = predict_absolute(
        model, inputs, states, bank_actions, target_mean, target_std, device
    )
    absolute_actor = predict_absolute(
        model, inputs, states, actor_actions, target_mean, target_std, device
    )
    bank_result = bank_metrics(absolute_bank, bank_costs, 0.1)
    actor_result = candidate_selection_metrics(absolute_actor, actor_costs)
    result = {
        "absolute_bank": bank_result,
        "absolute_actor_triplet": actor_result,
        "absolute_latest_vs_selected": delta_metrics(
            absolute_actor[:, 2] - absolute_actor[:, 1],
            actor_costs[:, (1, 2)],
        ),
        "absolute_latest_vs_initial": delta_metrics(
            absolute_actor[:, 2] - absolute_actor[:, 0],
            actor_costs[:, (0, 2)],
        ),
        "absolute_by_speed": {},
    }
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result["absolute_by_speed"][f"{value:.1f}"] = {
            "bank": bank_metrics(absolute_bank[mask], bank_costs[mask], 0.1),
            "actor_triplet": candidate_selection_metrics(
                absolute_actor[mask], actor_costs[mask]
            ),
            "latest_vs_initial": delta_metrics(
                absolute_actor[mask, 2] - absolute_actor[mask, 0],
                actor_costs[mask][:, (0, 2)],
            ),
        }
    pair_bank = pair_actor = None
    if model.pair_delta_enabled:
        pair_bank = predict_pair_scores(
            model, inputs, states, bank_actions, target_std, device
        )
        pair_actor = predict_pair_scores(
            model, inputs, states, actor_actions, target_std, device
        )
        latest_vs_selected = predict_pair_delta(
            model, inputs, states, actor_actions[:, 2], actor_actions[:, 1],
            target_std, device,
        )
        result.update({
            "pair_bank": candidate_selection_metrics(pair_bank, bank_costs),
            "pair_actor_triplet": candidate_selection_metrics(pair_actor, actor_costs),
            "pair_latest_vs_selected": delta_metrics(
                latest_vs_selected, actor_costs[:, (1, 2)]
            ),
            "pair_latest_vs_initial": delta_metrics(
                pair_actor[:, 2], actor_costs[:, (0, 2)]
            ),
        })
    return result, absolute_bank, absolute_actor, pair_bank, pair_actor


def load_or_create_heldout(args, data, folds, device):
    path = args.output_dir / "outer_heldout_actor_candidates.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    contract = json.loads((args.run_dir / "contract.json").read_text())
    outer_fold = int(contract["outer_fold"])
    heldout = np.flatnonzero(folds == outer_fold)
    if len(heldout) != 600:
        raise AssertionError(f"expected 600 outer-heldout states, got {len(heldout)}")
    normalization, _ = load_actor_normalization(args.base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    seeds = [int(value) for value in args.seeds.split(",")]
    actions = np.empty((len(seeds), 3, len(heldout), 8, 2), np.float32)
    costs = np.empty((len(seeds), 3, len(heldout)), np.float32)
    checkpoints = []
    for seed_position, seed in enumerate(seeds):
        initial_path = args.actor_root / f"a0_fold{outer_fold}_seed{seed}.pt"
        initial = load_actor(
            initial_path, outer_fold, seed, device, support_multiplier=3.0
        )[0]
        selected_path = args.run_dir / f"seed_{seed}" / "actor_selected.pt"
        latest_path = args.run_dir / f"seed_{seed}" / "actor_latest.pt"
        actors = (
            initial,
            load_actor_checkpoint(selected_path, device)[0],
            load_actor_checkpoint(latest_path, device)[0],
        )
        paths = (initial_path, selected_path, latest_path)
        for role_position, (actor, checkpoint) in enumerate(zip(actors, paths)):
            actor.eval()
            actions[seed_position, role_position] = actor_mean(
                actor, actor_inputs, heldout, device
            )
            costs[seed_position, role_position] = rollout_bank(
                backend, weights, params,
                actions[seed_position, role_position, :, None],
                states, current, references, heldout,
                args.rollout_batch_size, device,
            )[:, 0]
            checkpoints.append({
                "seed": seed, "role": ROLES[role_position],
                "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint),
            })
    result = {
        "state_index": heldout,
        "episode": data["episode"][heldout],
        "snapshot": data["snapshot"][heldout],
        "speed": data["speed"][heldout],
        "scenario": data["scenario"][heldout],
        "actions": actions,
        "costs": costs,
        "params_json": np.asarray(params_json),
        "weights_json": np.asarray(weights_json),
        "dbm_json": np.asarray(dbm_json),
    }
    np.savez_compressed(path, **result)
    (args.output_dir / "outer_heldout_actor_manifest.json").write_text(
        json.dumps({
            "outer_fold": outer_fold,
            "state_count": len(heldout),
            "roles": ROLES,
            "checkpoints": checkpoints,
            "action_sha256": array_digest(actions),
            "cost_sha256": array_digest(costs),
            "formal_validation_loaded": False,
            "test_loaded": False,
        }, indent=2) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    arms = args.arms.split(",")
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"arms must be a subset of {ARMS}")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds:
        raise ValueError("at least one seed is required")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    run_contract = json.loads((args.run_dir / "contract.json").read_text())
    outer_fold = int(run_contract["outer_fold"])
    train = np.flatnonzero(folds != outer_fold)
    heldout = np.flatnonzero(folds == outer_fold)
    heldout_actor = load_or_create_heldout(args, data, folds, device)
    heldout_state = heldout_actor["state_index"]
    if not np.array_equal(heldout_state, heldout):
        raise AssertionError("heldout actor state indices mismatch fold")

    source_replays = {
        seed: args.run_dir / f"seed_{seed}" / "actor_visited_replay.npz"
        for seed in seeds
    }
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC_CRITIC_CAPACITY_PAIRDELTA_AB_CONTRACT",
        "arguments": serialize_args(args),
        "outer_fold": outer_fold,
        "arms": arms,
        "train_states": int(len(train)),
        "outer_heldout_states": int(len(heldout)),
        "actor_constructed_for_heldout_candidate_generation_only": True,
        "actor_optimizer_constructed": False,
        "actor_update_count": 0,
        "same_batches_across_arms_per_seed": True,
        "target": "standardized log1p deterministic J_direct",
        "base_loss": "smooth-L1 absolute value + within-state ranking",
        "pair_delta_loss": (
            "base loss + antisymmetric same-state delta-log-value smooth-L1 "
            "+ pair-head ranking"
        ),
        "source_hashes": {
            "candidate_bank": sha256_file(args.bank_root / "candidate_bank.npz"),
            "run_contract": sha256_file(args.run_dir / "contract.json"),
            **{
                f"replay_seed_{seed}": sha256_file(path)
                for seed, path in source_replays.items()
            },
        },
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    records = []
    predictions: dict[str, np.ndarray] = {
        "heldout_state_index": heldout,
        "heldout_bank_cost": data["costs"][heldout],
        "heldout_actor_cost": heldout_actor["costs"],
        "heldout_speed": data["speed"][heldout],
    }
    for seed_position, seed in enumerate(seeds):
        with np.load(source_replays[seed], allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        if not np.all(np.isin(replay["state_index"], train)):
            raise AssertionError("outer-heldout leakage in fixed replay")
        training_payload = torch.load(
            args.run_dir / f"seed_{seed}" / "critic1.pt", map_location="cpu"
        )
        target_mean = float(training_payload["training"]["target_mean"])
        target_std = float(training_payload["training"]["target_std"])
        inputs = critic_state_inputs(data, training_payload)
        max_round = int(replay["round"].max())
        for arm in arms:
            # Resetting both RNGs makes the sampled point/pair batches exactly
            # matched across arms for a given seed.
            experiment_seed = 260825100 + seed
            set_seed(experiment_seed)
            rng = np.random.default_rng(experiment_seed)
            model = model_for_arm(arm).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            losses = []
            schedule_hash = hashlib.sha256()
            for update in range(1, args.updates + 1):
                points = sample_training_points(
                    data, train, replay, max_round,
                    args.batch_size, rng,
                )
                pairs = sample_pairs(
                    data, train, replay, args.pair_batch_size,
                    args.material_gap, rng,
                )
                for value in (*points, *pairs):
                    array = np.ascontiguousarray(value)
                    schedule_hash.update(str(array.dtype).encode())
                    schedule_hash.update(np.asarray(array.shape, np.int64).tobytes())
                    schedule_hash.update(array.tobytes())
                row = update_model(
                    model, optimizer, inputs, target_mean, target_std,
                    points, pairs, args, device,
                )
                losses.append(row)
                if update % args.report_every == 0 or update == args.updates:
                    recent = losses[-min(args.report_every, len(losses)):]
                    print(
                        f"seed={seed} arm={arm} update={update}/{args.updates} "
                        f"loss={np.mean([x['loss'] for x in recent]):.4f} "
                        f"value={np.mean([x['value'] for x in recent]):.4f} "
                        f"pair={np.mean([x['pair_delta'] for x in recent]):.4f}",
                        flush=True,
                    )
            evaluation, absolute_bank, absolute_actor, pair_bank, pair_actor = (
                evaluate_arm(
                    model, inputs, heldout,
                    data["actions"][heldout], data["costs"][heldout],
                    heldout_actor["actions"][seed_position].transpose(1, 0, 2, 3),
                    heldout_actor["costs"][seed_position].T,
                    data["speed"][heldout], target_mean, target_std, device,
                )
            )
            # load_or_create uses [seed,role,state,...]; evaluation expects
            # [state,role,...].  The explicit transpose above is part of the
            # saved contract and independently checked by the validator.
            parameter_count = int(sum(p.numel() for p in model.parameters()))
            arm_dir = args.output_dir / f"seed_{seed}" / arm
            arm_dir.mkdir(parents=True)
            torch.save({
                "model_class": "ConfigurableAbsoluteActionValueCritic",
                "arm": arm, "seed": seed, "outer_fold": outer_fold,
                "model": model.state_dict(), "model_config": model.config,
                "parameter_count": parameter_count,
                "training": training_payload["training"],
                "updates": args.updates,
                "actor_update_count": 0,
                "formal_validation_loaded": False,
                "test_loaded": False,
            }, arm_dir / "critic.pt")
            record = {
                "seed": seed, "arm": arm,
                "parameter_count": parameter_count,
                "model_config": model.config,
                "updates": args.updates,
                "training_schedule_sha256": schedule_hash.hexdigest(),
                "final_loss_mean_400": {
                    key: float(np.mean([row[key] for row in losses[-400:]]))
                    for key in losses[-1]
                },
                "evaluation": evaluation,
                "checkpoint": str((arm_dir / "critic.pt").resolve()),
            }
            (arm_dir / "summary.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            records.append(record)
            key = f"seed{seed}_{arm}"
            predictions[f"{key}_absolute_bank"] = absolute_bank
            predictions[f"{key}_absolute_actor"] = absolute_actor
            if pair_bank is not None:
                predictions[f"{key}_pair_bank"] = pair_bank
                predictions[f"{key}_pair_actor"] = pair_actor
            print(
                f"EVAL seed={seed} arm={arm} "
                f"bank_pair={evaluation['absolute_bank']['material_pair_accuracy']:.3f} "
                f"bank_regret={evaluation['absolute_bank']['top1_regret_mean']:.3f} "
                f"actor_regress_auc={evaluation['absolute_latest_vs_initial']['regress_auc']:.3f}",
                flush=True,
            )
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    aggregate: dict[str, Any] = {}
    for arm in arms:
        rows = [row for row in records if row["arm"] == arm]
        aggregate[arm] = {
            "parameter_count": rows[0]["parameter_count"],
            "heldout_bank_material_pair_accuracy": distribution(np.asarray([
                row["evaluation"]["absolute_bank"]["material_pair_accuracy"]
                for row in rows
            ])),
            "heldout_bank_top1_regret": distribution(np.asarray([
                row["evaluation"]["absolute_bank"]["top1_regret_mean"]
                for row in rows
            ])),
            "actor_latest_vs_initial_auc": distribution(np.asarray([
                row["evaluation"]["absolute_latest_vs_initial"]["regress_auc"]
                for row in rows
            ])),
            "actor_latest_vs_initial_severe_recall": distribution(np.asarray([
                row["evaluation"]["absolute_latest_vs_initial"]["severe_recall_at_zero"]
                for row in rows
            ])),
        }
        if arm == "pair_delta":
            aggregate[arm]["pair_actor_latest_vs_initial_auc"] = distribution(
                np.asarray([
                    row["evaluation"]["pair_latest_vs_initial"]["regress_auc"]
                    for row in rows
                ])
            )
            aggregate[arm]["pair_actor_latest_vs_initial_severe_recall"] = distribution(
                np.asarray([
                    row["evaluation"]["pair_latest_vs_initial"]["severe_recall_at_zero"]
                    for row in rows
                ])
            )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "contract": str((args.output_dir / "contract.json").resolve()),
        "records": records,
        "aggregate": aggregate,
        "actor_update_count": 0,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "aggregate": aggregate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
