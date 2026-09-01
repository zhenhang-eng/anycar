#!/usr/bin/env python3
"""Train a fixed-K no-anchor G-X Actor against the train-only J16 elite set.

This is a deliberately narrow multi-mode diagnostic.  K=1 and K=4 share the
same state encoder, knot-token transformer, output support and optimizer.  The
only functional differences are the number of final output heads and the
per-sample permutation-invariant set loss.  K=4 is evaluated both with an
oracle DBM best-of-K selector and with the independently trained absolute-value
Critic.  Formal validation and test artifacts are never read.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIContinuousCenterEncoder,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from mppi_a2_actors import KNOT_TIMES, knot_geometry, knot_sensitivity, knot_time_encoding
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
)
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs


DEFAULT_BANK = Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1")
DEFAULT_CRITIC = DEFAULT_BANK
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/j16_multi_candidate_actor_20260824_v1")
DEFAULT_GT_V1 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
BASE_SIGMA = np.asarray((0.25, 0.35), np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--critic-root", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-counts", default="1,4")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--elite-relative-gap", type=float, default=0.005)
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


def distribution(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, np.float64)
    return {
        "count": int(len(value)), "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)), "maximum": float(np.max(value)),
    }


class MultiCandidateGTXActor(nn.Module):
    """Clean no-anchor G-X trunk with K deterministic final candidate heads."""

    def __init__(
        self, candidate_count: int, center: torch.Tensor, scale: torch.Tensor,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self.candidate_count = int(candidate_count)
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        token_dim = 64
        # zero-anchor(2), time(2), sensitivity(1), G geometry(4), X(1)
        self.static_projection = nn.Linear(10, token_dim)
        self.feature_projection = nn.Linear(192, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.knot_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_heads = nn.ModuleList(
            nn.Linear(token_dim, 2) for _ in range(candidate_count)
        )
        self.output_skips = nn.ModuleList(
            nn.Linear(192, 16) for _ in range(candidate_count)
        )
        for module in (*self.output_heads, *self.output_skips):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.register_buffer("time_encoding", knot_time_encoding())
        self.register_buffer("sensitivity", knot_sensitivity())
        self.register_buffer("out_center", center.reshape(1, 1, 8, 2).clone())
        self.register_buffer("out_scale", scale.reshape(1, 1, 8, 2).clone())

    def encode_latent(self, history: torch.Tensor, reference: torch.Tensor,
                      current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(history)
        anchor = torch.zeros(batch, 8, 2, device=history.device, dtype=history.dtype)
        feedback = torch.zeros(batch, 74, device=history.device, dtype=history.dtype)
        gradient = torch.zeros(batch, 32, device=history.device, dtype=history.dtype)
        feature = self.encoder(history, reference, current, anchor, feedback, gradient)
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.sensitivity[None, :, None].expand(batch, -1, -1)
        static = torch.cat((
            anchor, times, sensitivity, knot_geometry(reference),
            reference[:, KNOT_TIMES, 0:1],
        ), dim=-1)
        tokens = self.static_projection(static) + self.feature_projection(feature)[:, None]
        decoded = self.knot_encoder(tokens)
        return feature, decoded

    def raw_from_latent(self, feature: torch.Tensor,
                        decoded: torch.Tensor) -> torch.Tensor:
        batch = len(feature)
        return torch.stack([
            head(decoded) + skip(feature).reshape(batch, 8, 2)
            for head, skip in zip(self.output_heads, self.output_skips)
        ], dim=1)

    def forward(self, history: torch.Tensor, reference: torch.Tensor,
                current: torch.Tensor) -> torch.Tensor:
        feature, decoded = self.encode_latent(history, reference, current)
        raw = self.raw_from_latent(feature, decoded)
        return torch.clamp(
            torch.tanh(raw) * self.out_scale + self.out_center, -1.0, 1.0
        )


def select_elite_set(data: dict[str, np.ndarray], count: int,
                     relative_gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = data["actions"][:, 16:24]
    costs = data["costs"][:, 16:24]
    selected_actions, selected_costs, unique_counts = [], [], []
    for state_actions, state_costs in zip(actions, costs):
        order = np.argsort(state_costs)
        best = float(state_costs[order[0]])
        eligible = [
            int(i) for i in order
            if float(state_costs[i]) <= best * (1.0 + relative_gap) + 1e-5
        ]
        chosen = [eligible[0]]
        while len(chosen) < min(count, len(eligible)):
            remaining = [i for i in eligible if i not in chosen]
            distance = []
            for index in remaining:
                delta = (
                    state_actions[index][None] - state_actions[chosen]
                ) / BASE_SIGMA
                distance.append(float(np.min(np.linalg.norm(delta.reshape(len(chosen), -1), axis=1))))
            chosen.append(remaining[int(np.argmax(distance))])
        unique_counts.append(len(chosen))
        while len(chosen) < count:
            chosen.append(chosen[0])
        selected_actions.append(state_actions[chosen])
        selected_costs.append(state_costs[chosen])
    return (
        np.asarray(selected_actions, np.float32),
        np.asarray(selected_costs, np.float32),
        np.asarray(unique_counts, np.int16),
    )


def matching_loss(predicted: torch.Tensor, target: torch.Tensor,
                  permutations: torch.Tensor) -> torch.Tensor:
    sigma = predicted.new_tensor(BASE_SIGMA).reshape(1, 1, 1, 1, 2)
    pair = ((predicted[:, :, None] - target[:, None]) / sigma).square().mean((-1, -2))
    heads = torch.arange(predicted.shape[1], device=predicted.device)
    # pair[:, heads, permutations] -> [B, number_of_permutations, K].
    assignment = pair[:, heads[None], permutations].mean(-1)
    return assignment.min(dim=1).values.mean()


def rollout_candidates(backend, weights, params, knots, states, current,
                       reference, batch_size, device) -> np.ndarray:
    result = []
    for begin in range(0, len(knots), batch_size):
        stop = min(begin + batch_size, len(knots))
        with torch.no_grad():
            action = interpolate_knots(
                torch.from_numpy(knots[begin:stop]).to(device), params.horizon
            )
            cost = batched_cost(
                backend, weights, action,
                torch.from_numpy(states[begin:stop]).to(device),
                torch.from_numpy(current[begin:stop]).to(device),
                torch.from_numpy(reference[begin:stop]).to(device),
            )
        result.append(cost.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def critic_select(critic_root: Path, seed: int, fold: int,
                  data: dict[str, np.ndarray], candidates: np.ndarray,
                  indices: np.ndarray, device: torch.device) -> np.ndarray:
    path = critic_root / f"critic_seed{seed}_fold{fold}.pt"
    payload = torch.load(path, map_location=device)
    normalizer = MPPIProposalNormalization.from_dict(payload["training"]["normalization"])
    history, reference, current = normalizer.normalize_numpy(
        data["history"][indices], data["reference"][indices], data["current"][indices]
    )
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for begin in range(0, len(indices), 256):
            local = slice(begin, begin + 256)
            value = model(
                torch.from_numpy(history[local].astype(np.float32)).to(device),
                torch.from_numpy(reference[local].astype(np.float32)).to(device),
                torch.from_numpy(current[local].astype(np.float32)).to(device),
                torch.from_numpy(candidates[local]).to(device),
            )
            predictions.append(value.cpu().numpy())
    return np.argmin(np.concatenate(predictions), axis=1)


def metrics(warm: np.ndarray, teacher: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    denominator = float(np.sum(warm - teacher))
    gain = warm - cost
    return {
        "cost": distribution(cost), "gain_vs_warm": distribution(gain),
        "headroom_recovery": float(np.sum(gain) / denominator),
        "regression_fraction": float(np.mean(gain < -1e-5)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    candidate_counts = [int(x) for x in args.candidate_counts.split(",")]
    if any(x not in (1, 4) for x in candidate_counts):
        raise ValueError("this registered pilot supports only K=1 and K=4")
    seeds = [int(x) for x in args.seeds.split(",")]
    device = torch.device(args.device)
    with np.load(args.bank_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    folds = make_folds(data, args.folds)
    states, current_action, direct_reference, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    best_index = np.argmin(data["costs"][:, 16:24], axis=1)
    j16 = data["costs"][:, 16:24][np.arange(len(data["costs"])), best_index]
    warm = data["costs"][:, 0]
    all_records: list[dict[str, Any]] = []
    oof_artifact: dict[int, dict[str, list[np.ndarray]]] = {
        count: {name: [] for name in (
            "seed", "fold", "state_index", "candidates", "candidate_cost",
            "oracle_selected_cost", "critic_selected_cost",
        )} for count in candidate_counts
    }
    args.output_dir.mkdir(parents=True)

    for candidate_count in candidate_counts:
        targets, target_costs, unique_counts = select_elite_set(
            data, candidate_count, args.elite_relative_gap
        )
        permutation = torch.tensor(
            list(itertools.permutations(range(candidate_count))),
            dtype=torch.long, device=device,
        )
        for seed in seeds:
            for fold in range(args.folds):
                set_seed(seed * 100 + fold)
                train = folds != fold
                heldout = folds == fold
                normalizer = MPPIProposalNormalization.fit(
                    data["history"][train], data["reference"][train], data["current"][train]
                )
                history, reference, current = normalizer.normalize_numpy(
                    data["history"], data["reference"], data["current"]
                )
                training_targets = targets[train].reshape(-1, 8, 2)
                center = torch.from_numpy(training_targets.mean(0)).to(device)
                scale = torch.from_numpy(training_targets.std(0) + 1e-6).to(device)
                model = MultiCandidateGTXActor(
                    candidate_count, center, scale, args.dropout
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6
                )
                rng = np.random.default_rng(260824 + seed * 100 + fold)
                train_index = np.flatnonzero(train)
                target_tensor = torch.from_numpy(targets).to(device)
                best_state, best_loss = None, float("inf")
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    losses = []
                    order = train_index[rng.permutation(len(train_index))]
                    for begin in range(0, len(train_index), args.batch_size):
                        index = order[begin:begin + args.batch_size]
                        index_t = torch.from_numpy(index).to(device)
                        predicted = model(
                            torch.from_numpy(history[index]).to(device),
                            torch.from_numpy(reference[index]).to(device),
                            torch.from_numpy(current[index]).to(device),
                        )
                        loss = matching_loss(predicted, target_tensor[index_t], permutation)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    scheduler.step()
                    mean_loss = float(np.mean(losses))
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        best_state = copy.deepcopy(model.state_dict())
                assert best_state is not None
                model.load_state_dict(best_state)
                model.eval()
                predicted_chunks = []
                with torch.no_grad():
                    for begin in range(0, len(data["history"]), 256):
                        local = slice(begin, begin + 256)
                        predicted_chunks.append(model(
                            torch.from_numpy(history[local]).to(device),
                            torch.from_numpy(reference[local]).to(device),
                            torch.from_numpy(current[local]).to(device),
                        ).cpu().numpy())
                predicted = np.concatenate(predicted_chunks).astype(np.float32)
                candidate_cost = rollout_candidates(
                    backend, weights, params, predicted, states, current_action,
                    direct_reference, args.evaluation_batch_size, device,
                )
                oracle_cost = candidate_cost.min(1)
                heldout_index = np.flatnonzero(heldout)
                critic_choice = critic_select(
                    args.critic_root, seed, fold, data, predicted[heldout],
                    heldout_index, device,
                )
                critic_cost = candidate_cost[heldout][np.arange(len(heldout_index)), critic_choice]
                record = {
                    "candidate_count": candidate_count, "seed": seed, "fold": fold,
                    "train": metrics(warm[train], j16[train], oracle_cost[train]),
                    "heldout_oracle_best_of_k": metrics(
                        warm[heldout], j16[heldout], oracle_cost[heldout]
                    ),
                    "heldout_critic_selected": metrics(
                        warm[heldout], j16[heldout], critic_cost
                    ),
                    "best_training_set_loss": best_loss,
                    "eligible_unique_target_count": distribution(unique_counts[train]),
                    "candidate_pair_distance_sigma": (
                        None if candidate_count == 1 else distribution(
                            np.linalg.norm(
                                ((predicted[:, :, None] - predicted[:, None]) / BASE_SIGMA)
                                .reshape(len(predicted), candidate_count, candidate_count, -1),
                                axis=-1,
                            )[:, np.triu_indices(candidate_count, 1)[0],
                              np.triu_indices(candidate_count, 1)[1]].reshape(-1)
                        )
                    ),
                }
                all_records.append(record)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "candidate_count": candidate_count, "seed": seed, "fold": fold,
                    "normalization": normalizer.to_dict(),
                    "heldout_episodes": sorted(np.unique(data["episode"][heldout]).tolist()),
                    "target_contract": "top J16 elites within 0.5%, diverse greedy, padded with best",
                }
                torch.save(
                    checkpoint,
                    args.output_dir / f"actor_k{candidate_count}_fold{fold}_seed{seed}.pt",
                )
                oof_artifact[candidate_count]["seed"].append(
                    np.full(len(heldout_index), seed, np.int8)
                )
                oof_artifact[candidate_count]["fold"].append(
                    np.full(len(heldout_index), fold, np.int8)
                )
                oof_artifact[candidate_count]["state_index"].append(heldout_index)
                oof_artifact[candidate_count]["candidates"].append(predicted[heldout])
                oof_artifact[candidate_count]["candidate_cost"].append(candidate_cost[heldout])
                oof_artifact[candidate_count]["oracle_selected_cost"].append(oracle_cost[heldout])
                oof_artifact[candidate_count]["critic_selected_cost"].append(critic_cost)
                print(
                    f"K={candidate_count} seed={seed} fold={fold} "
                    f"train_R={record['train']['headroom_recovery']:.3f} "
                    f"OOF_oracle_R={record['heldout_oracle_best_of_k']['headroom_recovery']:.3f} "
                    f"OOF_critic_R={record['heldout_critic_selected']['headroom_recovery']:.3f}"
                )

    saved_oof: dict[str, np.ndarray] = {"warm_cost": warm, "j16_cost": j16}
    for candidate_count, fields in oof_artifact.items():
        for key, values in fields.items():
            saved_oof[f"k{candidate_count}_{key}"] = np.concatenate(values)
    np.savez_compressed(args.output_dir / "oof_predictions.npz", **saved_oof)
    per_seed: dict[str, Any] = {}
    for candidate_count in candidate_counts:
        per_seed[str(candidate_count)] = {}
        for seed in seeds:
            rows = [r for r in all_records if r["candidate_count"] == candidate_count and r["seed"] == seed]
            per_seed[str(candidate_count)][str(seed)] = {
                "train_recovery_mean": float(np.mean([r["train"]["headroom_recovery"] for r in rows])),
                "oof_oracle_recovery_mean": float(np.mean([
                    r["heldout_oracle_best_of_k"]["headroom_recovery"] for r in rows
                ])),
                "oof_critic_recovery_mean": float(np.mean([
                    r["heldout_critic_selected"]["headroom_recovery"] for r in rows
                ])),
            }
    k1 = np.median([v["oof_oracle_recovery_mean"] for v in per_seed.get("1", {}).values()])
    k4 = np.median([v["oof_oracle_recovery_mean"] for v in per_seed.get("4", {}).values()])
    if 4 in candidate_counts and 1 in candidate_counts and k4 >= k1 + 0.10:
        decision = (
            "MULTI_CANDIDATE_ACTOR_ADDS_USABLE_OOF_HEADROOM"
            if k4 > 0.0
            else "MULTI_CANDIDATE_ACTOR_SMALL_RELATIVE_GAIN_BUT_STILL_UNUSABLE"
        )
    else:
        decision = "MULTI_CANDIDATE_ACTOR_NO_MATERIAL_OOF_GAIN"
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": decision,
        "contract": {
            "split": "train-only", "formal_validation_loaded": False,
            "test_loaded": False, "folds": args.folds, "seeds": seeds,
            "architecture": "shared clean/no-anchor G-X trunk; K final heads",
            "supervision": "permutation-invariant matched J16 elite set",
            "oracle_best_of_k_is_not_one-shot_deployable": True,
            "critic_selector": "episode-heldout absolute-action value Critic",
        },
        "sources": {
            "candidate_bank": str((args.bank_root / "candidate_bank.npz").resolve()),
            "candidate_bank_sha256": sha256_file(args.bank_root / "candidate_bank.npz"),
            "gt_v1_summary": str((args.gt_v1 / "summary.json").resolve()),
            "gt_v1_summary_sha256": sha256_file(args.gt_v1 / "summary.json"),
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "teacher": {
            "warm_cost": distribution(warm), "j16_cost": distribution(j16),
            "headroom": float(np.sum(warm - j16)),
        },
        "per_seed": per_seed,
        "median_seed_oof_oracle_recovery": {"K1": float(k1), "K4": float(k4)},
        "records": all_records,
        "decision_rule": (
            "relative gain requires K4 >= K1 + 0.10; usable success additionally "
            "requires K4 OOF recovery > 0"
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(decision)


if __name__ == "__main__":
    main()
