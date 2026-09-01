#!/usr/bin/env python3
"""Absolute-action scalar-value Critic audit on the sealed train split.

This is the final, deliberately narrow Critic experiment described after
review 11.82.  It does *not* regress an own-center gradient and it does not
receive a warm/anchor action.  The model predicts log(1 + J_direct(s, a)) for
an absolute 8x2 action-knot plan and is assessed as a value/ranking model.

Candidate bank per physical state (24 absolute plans):
  - GT v1: 8 heterogeneous raw starts;
  - GT v1: the corresponding 8 optimized centers;
  - GT v2: 8 continued/refined optimized centers.

All outer splits are episode grouped.  Formal validation/test are never read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TemporalConvEncoder,
    ego_reference_features,
)


DEFAULT_GT_V1 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
DEFAULT_GT_V2 = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2")
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/j16_oracle_labels_20260819_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"
)
KNOT_TIMES = (0, 7, 14, 21, 28, 35, 42, 49)
FLAT_SENSITIVITY = (
    1.031, 2.347, 1.660, 3.992, 1.431, 3.455, 1.120, 2.858,
    0.816, 2.190, 0.543, 1.478, 0.271, 0.708, 0.051, 0.121,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT_V1)
    parser.add_argument("--gt-v2", type=Path, default=DEFAULT_GT_V2)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage", choices=("dataset", "tiny", "cv", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--tiny-epochs", type=int, default=500)
    parser.add_argument("--tiny-states", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--material-gap", type=float, default=0.1)
    parser.add_argument(
        "--stationarity-weight", type=float, default=0.0,
        help=(
            "penalty on the physical log-value action-gradient at each "
            "state's bank-best candidate; zero reproduces V0"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    manifest = json.loads(args.manifest.read_text())
    metadata = {
        (row["episode"], row["snapshot"]): row for row in manifest["states"]
    }
    summaries = []
    for root in (args.gt_v1, args.gt_v2):
        payload = json.loads((root / "summary.json").read_text())
        if payload["split"] != "train":
            raise AssertionError(f"{root} is not the train split")
        summaries.append({(r["episode"], r["snapshot"]): r for r in payload["rows"]})
    keys = sorted(set(summaries[0]) & set(summaries[1]) & set(metadata))
    if len(keys) != 1800:
        raise AssertionError(f"expected 1800 shared train states, got {len(keys)}")

    fields: dict[str, list] = {name: [] for name in (
        "history", "reference", "current", "actions", "costs", "episode",
        "snapshot", "speed", "scenario", "source_sha256",
    )}
    for episode, snapshot in keys:
        rows = summaries[0][(episode, snapshot)], summaries[1][(episode, snapshot)]
        p1 = Path(rows[0]["result"])
        p2 = Path(rows[1]["result"])
        with np.load(p1, allow_pickle=False) as v1, np.load(p2, allow_pickle=False) as v2:
            if str(v1["source_sha256"]) != str(v2["source_sha256"]):
                raise AssertionError(f"source mismatch for {episode}/{snapshot}")
            actions = np.concatenate((
                np.asarray(v1["initial_knots"], np.float32),
                np.asarray(v1["optimized_knots"], np.float32),
                np.asarray(v2["optimized_knots"], np.float32),
            ))
            costs = np.concatenate((
                np.asarray(v1["initial_cost"], np.float32),
                np.asarray(v1["knot_cost_replay"], np.float32),
                np.asarray(v2["knot_cost_replay"], np.float32),
            ))
            source_path = Path(str(v1["source_snapshot"]))
        with np.load(source_path, allow_pickle=False) as source:
            state = np.asarray(source["initial_state"], np.float32)
            history = np.asarray(source["history"][0], np.float32)
            reference = ego_reference_features(source["reference_ego"], float(state[3]))
            current_action = np.asarray(source["current_action"], np.float32)
            current = np.asarray((state[3], state[4], *current_action), np.float32)
        if actions.shape != (24, 8, 2) or costs.shape != (24,):
            raise AssertionError(f"bad candidate bank shape for {episode}/{snapshot}")
        meta = metadata[(episode, snapshot)]
        fields["history"].append(history)
        fields["reference"].append(reference)
        fields["current"].append(current)
        fields["actions"].append(actions)
        fields["costs"].append(costs)
        fields["episode"].append(episode)
        fields["snapshot"].append(snapshot)
        fields["speed"].append(float(meta["speed"]))
        fields["scenario"].append(str(meta["scenario"]))
        fields["source_sha256"].append(str(meta["source_sha256"]))

    result = {
        "history": np.asarray(fields["history"], np.float32),
        "reference": np.asarray(fields["reference"], np.float32),
        "current": np.asarray(fields["current"], np.float32),
        "actions": np.asarray(fields["actions"], np.float32),
        "costs": np.asarray(fields["costs"], np.float32),
        "episode": np.asarray(fields["episode"]),
        "snapshot": np.asarray(fields["snapshot"]),
        "speed": np.asarray(fields["speed"], np.float32),
        "scenario": np.asarray(fields["scenario"]),
        "source_sha256": np.asarray(fields["source_sha256"]),
        "candidate_stage": np.asarray(
            ["raw_v1"] * 8 + ["optimized_v1"] * 8 + ["refined_v2"] * 8
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "candidate_bank.npz", **result)
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "TRAIN_ONLY_ABSOLUTE_ACTION_VALUE_CRITIC_CANDIDATE_BANK",
        "state_count": len(keys),
        "candidates_per_state": 24,
        "action_semantics": "absolute 8x2 knots [acceleration, steering]",
        "input_contract": "history + reference + current + candidate absolute action; no warm/anchor/residual",
        "candidate_stages": {"raw_v1": 8, "optimized_v1": 8, "refined_v2": 8},
        "hashes": {
            "gt_v1_summary": sha256_file(args.gt_v1 / "summary.json"),
            "gt_v2_summary": sha256_file(args.gt_v2 / "summary.json"),
            "manifest": sha256_file(args.manifest),
        },
        "formal_validation_test_policy": "not read; sealed",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return result


def load_or_build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    path = args.output_dir / "candidate_bank.npz"
    if not path.exists():
        return build_dataset(args)
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def knot_geometry(reference: torch.Tensor) -> torch.Tensor:
    samples = reference[:, KNOT_TIMES, :]
    lateral = samples[..., 1]
    heading = torch.atan2(samples[..., 2], samples[..., 3])
    speed = samples[..., 4]
    longitudinal = samples[..., 0]
    return torch.stack((longitudinal, lateral, heading, speed), dim=-1)


class AbsoluteActionValueCritic(nn.Module):
    """State-conditioned scalar J(s,a_abs), without an anchor action input."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(320, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU()
        )
        self.state_projection = nn.Linear(192, 64)
        # action(2), normalized time(2), population sensitivity(1), geometry(4)
        self.token_projection = nn.Linear(9, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.value_head = nn.Sequential(
            nn.Linear(64 + 192 + 16, 192), nn.SiLU(),
            nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        times = torch.tensor(KNOT_TIMES, dtype=torch.float32) / 49.0
        self.register_buffer("time_encoding", torch.stack((times, 1.0 - times), -1))
        sensitivity = torch.tensor(FLAT_SENSITIVITY).reshape(8, 2).sum(1)
        self.register_buffer("sensitivity", sensitivity / sensitivity.max())

    def forward(self, history, reference, current, actions):
        # State is encoded once; K absolute candidate plans are evaluated jointly.
        state = self.state_fusion(torch.cat((
            self.history_encoder(history), self.reference_encoder(reference),
            self.current_encoder(current),
        ), dim=-1))
        batch, candidates = actions.shape[:2]
        geometry = knot_geometry(reference)
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.sensitivity[None, :, None].expand(batch, -1, -1)
        static = torch.cat((actions, times[:, None].expand(-1, candidates, -1, -1),
                            sensitivity[:, None].expand(-1, candidates, -1, -1),
                            geometry[:, None].expand(-1, candidates, -1, -1)), -1)
        tokens = self.token_projection(static.reshape(batch * candidates, 8, 9))
        state_token = self.state_projection(state)[:, None, None, :].expand(
            -1, candidates, 8, -1
        ).reshape(batch * candidates, 8, 64)
        encoded = self.action_encoder(tokens + state_token).mean(1)
        state_expanded = state[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, 192
        )
        value = self.value_head(torch.cat((
            encoded, state_expanded, actions.reshape(batch * candidates, 16)
        ), -1))
        return value.reshape(batch, candidates)


def make_folds(data: dict[str, np.ndarray], folds: int) -> np.ndarray:
    cell_episodes: dict[tuple[float, str], list[str]] = {}
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        key = (round(float(data["speed"][mask][0]), 2), str(data["scenario"][mask][0]))
        cell_episodes.setdefault(key, []).append(str(episode))
    fold_of_episode = {}
    for key, episodes in sorted(cell_episodes.items()):
        members = sorted(episodes)
        if len(members) != folds:
            raise AssertionError(f"cell {key} has {len(members)} episodes, expected {folds}")
        for fold, episode in enumerate(members):
            fold_of_episode[episode] = fold
    return np.asarray([fold_of_episode[str(x)] for x in data["episode"]], np.int64)


def normalize_data(data: dict[str, np.ndarray], train: np.ndarray):
    normalizer = MPPIProposalNormalization.fit(
        data["history"][train], data["reference"][train], data["current"][train]
    )
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    return history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32), normalizer


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def metrics(pred: np.ndarray, costs: np.ndarray, material_gap: float) -> dict:
    truth = np.log1p(costs)
    pair_correct = []
    pair_large = []
    for p, c in zip(pred, costs):
        i, j = np.triu_indices(len(c), 1)
        gap = np.abs(c[i] - c[j])
        correct = np.sign(p[i] - p[j]) == np.sign(c[i] - c[j])
        pair_correct.extend(correct[gap >= material_gap].tolist())
        pair_large.extend(correct[gap >= 1.0].tolist())
    selected = np.argmin(pred, axis=1)
    top1 = costs[np.arange(len(costs)), selected]
    top4_indices = np.argpartition(pred, 4, axis=1)[:, :4]
    top4 = np.take_along_axis(costs, top4_indices, axis=1).min(1)
    warm = costs[:, 0]
    best = costs.min(1)
    denominator = float(np.sum(warm - best))
    recovery_top1 = float(np.sum(warm - top1) / denominator)
    recovery_top4 = float(np.sum(warm - top4) / denominator)
    return {
        "count": int(len(costs)),
        "pearson_log_value": correlation(pred.ravel(), truth.ravel()),
        "spearman_log_value": correlation(rankdata(pred.ravel()), rankdata(truth.ravel())),
        "material_pair_accuracy": float(np.mean(pair_correct)),
        "large_gap_pair_accuracy": float(np.mean(pair_large)),
        "top1_cost_mean": float(np.mean(top1)),
        "top1_regret_mean": float(np.mean(top1 - best)),
        "top1_headroom_recovery": recovery_top1,
        "top4_cost_mean": float(np.mean(top4)),
        "top4_regret_mean": float(np.mean(top4 - best)),
        "top4_headroom_recovery": recovery_top4,
        "warm_cost_mean": float(np.mean(warm)),
        "oracle_cost_mean": float(np.mean(best)),
        "harmful_top1_fraction": float(np.mean(top1 > warm + 1e-5)),
        "near_best_top1_fraction": float(np.mean(top1 <= best * 1.005 + 1e-5)),
    }


def ranking_loss(pred: torch.Tensor, raw_cost: torch.Tensor, gap: float, temperature: float):
    candidates = pred.shape[1]
    i, j = torch.triu_indices(candidates, candidates, 1, device=pred.device)
    target_delta = raw_cost[:, i] - raw_cost[:, j]
    prediction_delta = pred[:, i] - pred[:, j]
    mask = target_delta.abs() >= gap
    return F.softplus(-target_delta[mask].sign() * prediction_delta[mask] / temperature).mean()


def train_model(args, data, train_indices, evaluation_indices, seed, epochs):
    set_seed(seed)
    device = torch.device(args.device)
    if args.stationarity_weight > 0 and device.type == "cuda":
        # PyTorch 2.3 efficient SDP has no double backward, which is required
        # when a loss is applied to dQ/da.  Math SDP preserves the same model
        # function and provides the missing second derivative.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    train_mask = np.zeros(len(data["costs"]), bool)
    train_mask[train_indices] = True
    history, reference, current, normalizer = normalize_data(data, train_mask)
    target_all = np.log1p(data["costs"]).astype(np.float32)
    target_mean = float(target_all[train_indices].mean())
    target_std = float(target_all[train_indices].std() + 1e-6)
    target_z = (target_all - target_mean) / target_std
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(seed)
    final_loss = None
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(train_indices)
        losses, stationarity_losses = [], []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            h = torch.from_numpy(history[index]).to(device)
            r = torch.from_numpy(reference[index]).to(device)
            c = torch.from_numpy(current[index]).to(device)
            a = torch.from_numpy(data["actions"][index]).to(device)
            y = torch.from_numpy(target_z[index]).to(device)
            raw = torch.from_numpy(data["costs"][index]).to(device)
            prediction = model(h, r, c, a)
            value = F.smooth_l1_loss(prediction, y)
            rank = ranking_loss(
                prediction, raw, args.material_gap, args.ranking_temperature
            )
            stationarity = prediction.new_zeros(())
            if args.stationarity_weight > 0:
                best_index = raw.argmin(dim=1)
                best_action = a[
                    torch.arange(len(a), device=device), best_index
                ][:, None].detach().clone().requires_grad_(True)
                best_prediction = model(h, r, c, best_action)
                physical_log_gradient = torch.autograd.grad(
                    (best_prediction * target_std).sum(), best_action,
                    create_graph=True,
                )[0]
                # Mean component energy keeps lambda independent of 16-D size.
                stationarity = physical_log_gradient.square().mean()
            loss = (
                value + args.ranking_weight * rank
                + args.stationarity_weight * stationarity
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            stationarity_losses.append(float(stationarity.detach()))
        final_loss = float(np.mean(losses))
        if epochs >= 200 and (epoch + 1) % 100 == 0:
            print(f"tiny epoch={epoch + 1} loss={final_loss:.5f}", flush=True)

    def predict(indices):
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(indices), 128):
                index = indices[start:start + 128]
                values.append(model(
                    torch.from_numpy(history[index]).to(device),
                    torch.from_numpy(reference[index]).to(device),
                    torch.from_numpy(current[index]).to(device),
                    torch.from_numpy(data["actions"][index]).to(device),
                ).cpu().numpy() * target_std + target_mean)
        return np.concatenate(values)

    return model, predict(np.asarray(train_indices)), predict(np.asarray(evaluation_indices)), {
        "final_loss": final_loss,
        "target_mean": target_mean,
        "target_std": target_std,
        "stationarity_weight": float(args.stationarity_weight),
        "final_stationarity_component_mse": float(np.mean(stationarity_losses)),
        "normalization": normalizer.to_dict(),
    }


def run_tiny(args, data) -> dict:
    # Stratified deterministic spread, not just consecutive frames.
    indices = np.linspace(0, len(data["costs"]) - 1, args.tiny_states, dtype=np.int64)
    model, train_pred, _, training = train_model(
        args, data, indices, indices, seed=0, epochs=args.tiny_epochs
    )
    result = metrics(train_pred, data["costs"][indices], args.material_gap)
    result.update(training)
    result["states"] = indices.tolist()
    result["gate"] = {
        "pearson_ge_0_98": result["pearson_log_value"] >= 0.98,
        "ranking_ge_0_98": result["material_pair_accuracy"] >= 0.98,
        "top1_recovery_ge_0_99": result["top1_headroom_recovery"] >= 0.99,
    }
    result["passed"] = bool(all(result["gate"].values()))
    torch.save({"model": model.state_dict(), "result": result}, args.output_dir / "tiny_overfit.pt")
    (args.output_dir / "tiny_overfit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_cv(args, data) -> dict:
    folds = make_folds(data, args.folds)
    seeds = [int(value) for value in args.seeds.split(",")]
    records = []
    for seed in seeds:
        for fold in range(args.folds):
            train = np.flatnonzero(folds != fold)
            heldout = np.flatnonzero(folds == fold)
            print(f"cv seed={seed} fold={fold} train={len(train)} heldout={len(heldout)}", flush=True)
            model, train_pred, heldout_pred, training = train_model(
                args, data, train, heldout, seed, args.epochs
            )
            train_metrics = metrics(train_pred, data["costs"][train], args.material_gap)
            oof_metrics = metrics(heldout_pred, data["costs"][heldout], args.material_gap)
            speed_metrics = {
                f"{speed:.1f}": metrics(
                    heldout_pred[data["speed"][heldout] == speed],
                    data["costs"][heldout][data["speed"][heldout] == speed],
                    args.material_gap,
                ) for speed in sorted(np.unique(data["speed"][heldout]))
            }
            checkpoint = args.output_dir / f"critic_seed{seed}_fold{fold}.pt"
            torch.save({
                "model_class": "AbsoluteActionValueCritic",
                "model": model.state_dict(), "seed": seed, "fold": fold,
                "training": training,
                "input_contract": "history+reference+current+absolute_action; no anchor",
            }, checkpoint)
            records.append({
                "seed": seed, "fold": fold, "training": training,
                "train": train_metrics, "oof": oof_metrics,
                "oof_by_speed": speed_metrics, "checkpoint": str(checkpoint),
            })
            print(
                f"  train rank={train_metrics['material_pair_accuracy']:.3f} "
                f"top4R={train_metrics['top4_headroom_recovery']:.3f}; "
                f"OOF rank={oof_metrics['material_pair_accuracy']:.3f} "
                f"top1R={oof_metrics['top1_headroom_recovery']:.3f} "
                f"top4R={oof_metrics['top4_headroom_recovery']:.3f}", flush=True,
            )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_STRICT_POOLED_ANALYSIS",
        "contract": {
            "target": "standardized log1p deterministic J_direct",
            "loss": "smooth-L1 value + within-state material-pair ranking",
            "stationarity_weight": float(args.stationarity_weight),
            "input": "state/reference/current + full absolute 8x2 candidate; no anchor/residual",
            "split": "3-fold episode grouped, 3 seeds, fixed epochs",
            "candidate_bank": "8 raw v1 + 8 optimized v1 + 8 refined v2",
            "gradient_use": "none; gradient audit only if value/ranking gate passes",
            "formal_validation_test": "sealed and not read",
        },
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return analyze_existing(args, data)


def predict_checkpoint(args, data, indices, checkpoint: Path) -> np.ndarray:
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location=device)
    training = payload["training"]
    normalizer = MPPIProposalNormalization.from_dict(training["normalization"])
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            index = indices[start:start + 128]
            pred = model(
                torch.from_numpy(history[index].astype(np.float32)).to(device),
                torch.from_numpy(reference[index].astype(np.float32)).to(device),
                torch.from_numpy(current[index].astype(np.float32)).to(device),
                torch.from_numpy(data["actions"][index]).to(device),
            )
            pred = pred * float(training["target_std"]) + float(training["target_mean"])
            values.append(pred.cpu().numpy())
    return np.concatenate(values)


def recovery_bootstrap(
    pred: np.ndarray, costs: np.ndarray, episodes: np.ndarray, seed: int,
    repetitions: int = 2000,
) -> dict[str, list[float]]:
    selected = np.argmin(pred, axis=1)
    top1 = costs[np.arange(len(costs)), selected]
    top4_index = np.argpartition(pred, 4, axis=1)[:, :4]
    top4 = np.take_along_axis(costs, top4_index, axis=1).min(1)
    warm, best = costs[:, 0], costs.min(1)
    unique = np.unique(episodes)
    by_episode = {episode: np.flatnonzero(episodes == episode) for episode in unique}
    rng = np.random.default_rng(seed + 71021)
    samples = {"top1": [], "top4": []}
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([by_episode[episode] for episode in chosen])
        denominator = np.sum(warm[index] - best[index])
        samples["top1"].append(float(np.sum(warm[index] - top1[index]) / denominator))
        samples["top4"].append(float(np.sum(warm[index] - top4[index]) / denominator))
    return {
        name: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for name, value in samples.items()
    }


def candidate_bank_diagnostics(
    pred: np.ndarray, costs: np.ndarray, material_gap: float, seed: int,
) -> dict:
    subsets = {
        "raw_v1": np.arange(0, 8),
        "optimized_v1": np.arange(8, 16),
        "refined_v2": np.arange(16, 24),
        "all_optimized": np.arange(8, 24),
    }
    stage_metrics = {
        name: metrics(pred[:, index], costs[:, index], material_gap)
        for name, index in subsets.items()
    }
    stage_choice = np.bincount(np.argmin(pred, axis=1) // 8, minlength=3)
    stage_choice = (stage_choice / len(pred)).tolist()
    warm, best = costs[:, 0], costs.min(1)
    denominator = np.sum(warm - best)
    rng = np.random.default_rng(seed + 982451653)
    random_baseline = {}
    for count in (1, 2, 4, 8):
        recovery, regret, harmful = [], [], []
        for _ in range(2000):
            random_score = rng.random(costs.shape)
            chosen = np.argpartition(random_score, count - 1, axis=1)[:, :count]
            chosen_cost = np.take_along_axis(costs, chosen, axis=1).min(1)
            recovery.append(float(np.sum(warm - chosen_cost) / denominator))
            regret.append(float(np.mean(chosen_cost - best)))
            harmful.append(float(np.mean(chosen_cost > warm + 1e-5)))
        random_baseline[f"random_top{count}"] = {
            "headroom_recovery_mean": float(np.mean(recovery)),
            "headroom_recovery_95_interval": [
                float(np.quantile(recovery, 0.025)),
                float(np.quantile(recovery, 0.975)),
            ],
            "regret_mean": float(np.mean(regret)),
            "harmful_fraction_mean": float(np.mean(harmful)),
        }
    return {
        "within_stage": stage_metrics,
        "selected_stage_fraction": dict(zip(
            ("raw_v1", "optimized_v1", "refined_v2"), stage_choice
        )),
        "random_shortlist_baseline": random_baseline,
        "interpretation": (
            "Top-k is partially easy because 16/24 candidates are optimized; "
            "within-stage ranking and the matched random-top-k baseline bound "
            "the stage-classification shortcut."
        ),
    }


def analyze_existing(args, data) -> dict:
    summary_path = args.output_dir / "summary.json"
    result = json.loads(summary_path.read_text())
    folds = make_folds(data, args.folds)
    seeds = [int(value) for value in args.seeds.split(",")]
    strict = []
    prediction_artifact = {}
    for seed in seeds:
        oof = np.empty_like(data["costs"], dtype=np.float32)
        for fold in range(args.folds):
            heldout = np.flatnonzero(folds == fold)
            checkpoint = args.output_dir / f"critic_seed{seed}_fold{fold}.pt"
            oof[heldout] = predict_checkpoint(args, data, heldout, checkpoint)
        prediction_artifact[f"seed_{seed}"] = oof
        pooled = metrics(oof, data["costs"], args.material_gap)
        pooled["episode_bootstrap_95ci"] = recovery_bootstrap(
            oof, data["costs"], data["episode"], seed
        )
        pooled["by_speed"] = {
            f"{speed:.1f}": metrics(
                oof[data["speed"] == speed], data["costs"][data["speed"] == speed],
                args.material_gap,
            ) for speed in sorted(np.unique(data["speed"]))
        }
        pooled["by_scenario"] = {
            str(scenario): metrics(
                oof[data["scenario"] == scenario],
                data["costs"][data["scenario"] == scenario], args.material_gap,
            ) for scenario in sorted(np.unique(data["scenario"]))
        }
        pooled["candidate_bank_diagnostics"] = candidate_bank_diagnostics(
            oof, data["costs"], args.material_gap, seed
        )
        strict.append({"seed": seed, "strict_pooled_oof": pooled})
    np.savez_compressed(
        args.output_dir / "strict_oof_predictions.npz", folds=folds,
        **prediction_artifact,
    )
    gates = []
    for row in strict:
        m = row["strict_pooled_oof"]
        gate = {
            "material_pair_accuracy_ge_0_80": m["material_pair_accuracy"] >= 0.80,
            "top4_headroom_recovery_ge_0_95": m["top4_headroom_recovery"] >= 0.95,
            "top1_aggregate_nonnegative": m["top1_headroom_recovery"] >= 0.0,
            "harmful_top1_fraction_le_0_10": m["harmful_top1_fraction"] <= 0.10,
        }
        gate["passed"] = bool(all(gate.values()))
        gates.append({"seed": row["seed"], **gate})
    passed = sum(int(row["passed"]) for row in gates)
    result.update({
        "qualification": (
            "ABSOLUTE_ACTION_VALUE_CRITIC_USABLE_FOR_RANKING"
            if passed >= 2 else "ABSOLUTE_ACTION_VALUE_CRITIC_NOT_USABLE"
        ),
        "strict_per_seed": strict,
        "gates": gates,
        "passed_seed_count": passed,
        "analysis_note": "All headline metrics are strict pooled OOF; fold means are diagnostic only.",
    })
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_build_dataset(args)
    print(
        f"dataset states={len(data['costs'])} candidates={data['costs'].shape[1]} "
        f"mean_cost={data['costs'].mean():.3f}", flush=True,
    )
    if args.stage == "dataset":
        return
    if args.stage in ("tiny", "all"):
        tiny = run_tiny(args, data)
        print("tiny", json.dumps({k: tiny[k] for k in (
            "pearson_log_value", "material_pair_accuracy",
            "top1_headroom_recovery", "passed")}, indent=2), flush=True)
        if not tiny["passed"] and args.stage == "all":
            raise SystemExit("tiny-set gate failed; refusing full CV")
    if args.stage in ("cv", "all"):
        result = run_cv(args, data)
        print("qualification", result["qualification"], flush=True)
    elif args.stage == "analyze":
        result = analyze_existing(args, data)
        print("qualification", result["qualification"], flush=True)


if __name__ == "__main__":
    main()
