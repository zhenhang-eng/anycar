#!/usr/bin/env python3
"""Train-only strict no-anchor Actor and state-balanced Twin Query Critics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from build_query_expected_road_fullrank_sidecar import interpolate_knots  # noqa: E402
from mppi_a2_actors import DirectNoAnchorGTXActor  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_single_center_oac_config_20260902_v1.json"
PAIR_SAMPLES_PER_STATE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
        "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if len(left) < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_data(config: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    source = Path(config["outputs"]["absolute_replay"]).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    if validation["qualification"] not in {
        "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
        "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
    }:
        raise AssertionError("absolute Replay did not independently pass")
    if sha256(source / "replay.npz") != manifest["replay_sha256"]:
        raise AssertionError("absolute Replay hash mismatch")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("formal validation/test was consumed")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("DBM fields or labels were consumed")
    with np.load(source / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    landscape = Path(manifest["landscape_source"])
    with np.load(landscape / "landscape.npz", allow_pickle=False) as archive:
        land = {name: np.asarray(archive[name]) for name in archive.files}
    branch = int(np.flatnonzero(land["branch_name"] == "observable_current_action_to_zero")[0])
    target = data["fullrank_teacher_knots"].copy()
    target_cost = data["fullrank_teacher_cost"].copy()
    mapped = np.flatnonzero(data["landscape_context_mask"])
    audit = data["landscape_source_audit_row"][mapped]
    target[mapped] = land["center_knots"][audit, branch, -1]
    target_cost[mapped] = land["center_cost"][audit, branch, -1]
    prior_branch = int(np.flatnonzero(land["branch_name"] == "prior_noanchor_canonical")[0])
    prior_cost = np.full(len(data["state"]), np.nan, np.float32)
    prior_cost[mapped] = land["center_cost"][audit, prior_branch, 0]
    data.update(
        {
            "actor_target_knots": target.astype(np.float32),
            "actor_target_cost": target_cost.astype(np.float32),
            "landscape_prior_cost": prior_cost,
            "source_dir": np.asarray(str(source)),
            "source_replay_sha256": np.asarray(manifest["replay_sha256"]),
        }
    )
    fullrank_manifest = json.loads((Path(manifest["fullrank_source"]) / "manifest.json").read_text())
    parent = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection = json.loads((Path(parent["source_collection"]) / "manifest.json").read_text())
    return data, manifest, collection


def normalized_inputs(data: dict[str, np.ndarray], fit: np.ndarray) -> tuple[tuple[np.ndarray, ...], MPPIProposalNormalization]:
    normalizer = MPPIProposalNormalization.fit(
        data["history"][fit], data["critic_reference"][fit], data["critic_current"][fit]
    )
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["critic_reference"], data["critic_current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    ), normalizer


def actor_center_scale(labels: np.ndarray, fit: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    center = labels[fit].mean(0).astype(np.float32)
    scale = (3.0 * labels[fit].std(0) + 1e-4).astype(np.float32)
    scale = np.maximum(scale, np.asarray((0.12, 0.12), np.float32)[None])
    return torch.from_numpy(center), torch.from_numpy(scale)


def actor_predict(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
                  device: torch.device, batch_size: int = 256) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            local = rows[start:start + batch_size]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
            output.append(model(*tensors)[1].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def query_cost(controller: TorchMPPIController, data: dict[str, np.ndarray],
               knots: np.ndarray, rows: np.ndarray) -> np.ndarray:
    sequences = interpolate_knots(knots)
    output = np.empty(len(rows), np.float32)
    for local, row in enumerate(rows):
        result = controller.evaluate_action_sequences(
            data["state"][row], data["current_action"][row],
            data["history"][row:row + 1], data["reference"][row],
            sequences[local:local + 1],
        )
        output[local] = float(result["cost"][0].cpu())
    return output


def actor_metrics(cost: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    gain = baseline.astype(np.float64) - cost.astype(np.float64)
    available = baseline.astype(np.float64) - target.astype(np.float64)
    material = available > 1e-5
    return {
        "cost": distribution(cost), "warm_relative_gain": distribution(gain),
        "target_gain": distribution(available),
        "gain_recovery": float(np.sum(gain[material]) / np.sum(available[material])) if np.any(material) else 0.0,
        "regression_fraction": float(np.mean(gain < -1e-5)),
    }


def train_actor(data: dict[str, np.ndarray], inputs: tuple[np.ndarray, ...], fit: np.ndarray,
                selection: np.ndarray, seed: int, config: dict, controller: TorchMPPIController,
                device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seed(seed)
    center, scale = actor_center_scale(data["actor_target_knots"], fit)
    actor = DirectNoAnchorGTXActor(dropout=0.0, center=center.to(device), scale=scale.to(device)).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-6)
    target = torch.from_numpy(data["actor_target_knots"]).to(device)
    epochs = int(config["actor"]["coarse_initialization"]["epochs"])
    stride = int(config["actor"]["coarse_initialization"]["selection_stride"])
    rng = np.random.default_rng(806_301 + seed)
    best_score, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        actor.train()
        losses = []
        order = rng.permutation(fit)
        for start in range(0, len(fit), 32):
            rows = order[start:start + 32]
            tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
            prediction = actor(*tensors)[1]
            loss = (prediction - target[rows]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % stride == 0 or epoch == epochs:
            prediction = actor_predict(actor, inputs, selection, device)
            cost = query_cost(controller, data, prediction, selection)
            score = float(np.mean(cost))
            history.append({"epoch": epoch, "fit_mse": float(np.mean(losses)), "selection_cost_mean": score})
            if score < best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(actor.state_dict())
    if best_state is None:
        raise AssertionError("Actor selection failed")
    actor.load_state_dict(best_state, strict=True)
    return actor, {
        "best_epoch": best_epoch, "best_selection_cost_mean": best_score,
        "selection_history": history, "out_center": center.tolist(), "out_scale": scale.tolist(),
    }


def state_balanced_target_stats(cost: np.ndarray, valid: np.ndarray, fit: np.ndarray) -> tuple[float, float]:
    means, squares = [], []
    for row in fit:
        values = np.log1p(cost[row, valid[row]].astype(np.float64))
        means.append(values.mean())
        squares.append(np.square(values).mean())
    mean = float(np.mean(means))
    std = float(max(np.sqrt(np.mean(squares) - mean * mean), 1e-6))
    return mean, std


def choose_candidates(data: dict[str, np.ndarray], rows: np.ndarray, rng: np.random.Generator,
                      count: int) -> np.ndarray:
    output = np.empty((len(rows), count), np.int64)
    for local, row in enumerate(rows):
        valid = np.flatnonzero(data["candidate_valid_mask"][row])
        cost = data["candidate_cost"][row, valid]
        chosen = [0, int(valid[np.argmin(cost)])]
        remaining = valid[~np.isin(valid, chosen)]
        uniform_count = min(23, len(remaining))
        chosen.extend(rng.choice(remaining, uniform_count, replace=False).tolist())
        remaining = valid[~np.isin(valid, chosen)]
        need = count - len(chosen)
        if need:
            order = remaining[np.argsort(data["candidate_cost"][row, remaining])]
            positions = np.linspace(0, len(order) - 1, need + 2)[1:-1]
            jitter = rng.uniform(-0.35, 0.35, size=need) * max(len(order) / max(need, 1), 1.0)
            indices = np.clip(np.rint(positions + jitter).astype(int), 0, len(order) - 1)
            for index in indices:
                candidate = int(order[index])
                if candidate not in chosen:
                    chosen.append(candidate)
            if len(chosen) < count:
                remaining = valid[~np.isin(valid, chosen)]
                chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
        output[local] = np.asarray(chosen[:count])
    return output


def critic_predict(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], data: dict[str, np.ndarray],
                   rows: np.ndarray, device: torch.device) -> np.ndarray:
    output = np.full((len(rows), data["candidate_knots"].shape[1]), np.nan, np.float32)
    model.eval()
    with torch.no_grad():
        for count in np.unique(data["candidate_count"][rows]):
            local_rows = np.flatnonzero(data["candidate_count"][rows] == count)
            state_batch = 12 if int(count) == 132 else 2
            for start in range(0, len(local_rows), state_batch):
                positions = local_rows[start:start + state_batch]
                global_rows = rows[positions]
                prediction = model(
                    torch.from_numpy(inputs[0][global_rows]).to(device),
                    torch.from_numpy(inputs[1][global_rows]).to(device),
                    torch.from_numpy(inputs[2][global_rows]).to(device),
                    torch.from_numpy(data["candidate_knots"][global_rows, :int(count)]).to(device),
                )
                output[positions, :int(count)] = prediction.cpu().numpy()
    return output


def state_metrics(predicted_log: np.ndarray, data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, Any]:
    correlations, accuracies, rmses = [], [], []
    selected_cost, oracle_cost = [], []
    for local, row in enumerate(rows):
        valid = data["candidate_valid_mask"][row]
        prediction = predicted_log[local, valid].astype(np.float64)
        truth = np.log1p(data["candidate_cost"][row, valid].astype(np.float64))
        correlations.append(correlation(prediction, truth))
        rmses.append(float(np.sqrt(np.mean(np.square(prediction - truth)))))
        rng = np.random.default_rng(1_234_000 + int(data["row_index"][row]))
        left = rng.integers(0, len(truth), PAIR_SAMPLES_PER_STATE)
        right = rng.integers(0, len(truth), PAIR_SAMPLES_PER_STATE)
        true_delta = truth[left] - truth[right]
        pred_delta = prediction[left] - prediction[right]
        material = np.abs(true_delta) > 1e-7
        accuracies.append(float(np.mean(np.sign(pred_delta[material]) == np.sign(true_delta[material]))))
        selected_cost.append(float(data["candidate_cost"][row, np.flatnonzero(valid)[np.argmin(prediction)]]))
        oracle_cost.append(float(np.min(data["candidate_cost"][row, valid])))
    return {
        "state_pearson_log_cost": distribution(np.asarray(correlations)),
        "state_pair_sign_accuracy": distribution(np.asarray(accuracies)),
        "state_log_rmse": distribution(np.asarray(rmses)),
        "selected_cost": distribution(np.asarray(selected_cost)),
        "oracle_cost": distribution(np.asarray(oracle_cost)),
    }


def landscape_gain_recovery(predicted_log: np.ndarray, data: dict[str, np.ndarray],
                            rows: np.ndarray) -> float:
    gains, available = [], []
    for local, row in enumerate(rows):
        if not data["landscape_context_mask"][row]:
            continue
        eligible = data["candidate_valid_mask"][row] & data["candidate_canonical_eligible_mask"][row]
        indices = np.flatnonzero(eligible)
        selected = indices[np.argmin(predicted_log[local, eligible])]
        best = indices[np.argmin(data["candidate_cost"][row, eligible])]
        baseline = float(data["landscape_prior_cost"][row])
        gains.append(baseline - float(data["candidate_cost"][row, selected]))
        available.append(baseline - float(data["candidate_cost"][row, best]))
    material = np.asarray(available) > 1e-5
    return float(np.sum(np.asarray(gains)[material]) / np.sum(np.asarray(available)[material])) if np.any(material) else 0.0


def critic_selection_score(metrics: dict[str, Any], recovery: float) -> float:
    return (
        metrics["state_log_rmse"]["median"]
        + 0.5 * (1.0 - metrics["state_pearson_log_cost"]["median"])
        + 0.5 * (1.0 - metrics["state_pair_sign_accuracy"]["median"])
        + max(0.0, -recovery)
    )


def train_critic(data: dict[str, np.ndarray], inputs: tuple[np.ndarray, ...], fit: np.ndarray,
                 selection: np.ndarray, seed: int, config: dict, device: torch.device
                 ) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seed(seed)
    critic_config = config["critic"]
    model = ConfigurableAbsoluteActionValueCritic().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(critic_config["learning_rate"]),
                                  weight_decay=float(critic_config["weight_decay"]))
    target_mean, target_std = state_balanced_target_stats(
        data["candidate_cost"], data["candidate_valid_mask"], fit
    )
    rng = np.random.default_rng(907_501 + seed)
    epochs = int(critic_config["epochs"])
    state_batch = int(critic_config["state_batch_size"])
    candidate_count = int(critic_config["candidates_per_state"])
    best_score, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), state_batch):
            rows = order[start:start + state_batch]
            chosen = choose_candidates(data, rows, rng, candidate_count)
            actions = torch.from_numpy(data["candidate_knots"][rows[:, None], chosen]).to(device)
            raw_cost = data["candidate_cost"][rows[:, None], chosen].astype(np.float64)
            truth_np = ((np.log1p(raw_cost) - target_mean) / target_std).astype(np.float32)
            truth = torch.from_numpy(truth_np).to(device)
            prediction = model(
                torch.from_numpy(inputs[0][rows]).to(device),
                torch.from_numpy(inputs[1][rows]).to(device),
                torch.from_numpy(inputs[2][rows]).to(device), actions,
            )
            value_loss = torch.nn.functional.smooth_l1_loss(prediction, truth)
            left = torch.randint(candidate_count, (len(rows), candidate_count), device=device)
            right = torch.randint(candidate_count, (len(rows), candidate_count), device=device)
            batch = torch.arange(len(rows), device=device)[:, None]
            pred_delta = prediction[batch, left] - prediction[batch, right]
            true_delta = truth[batch, left] - truth[batch, right]
            material = true_delta.abs() > 1e-5
            ranking = torch.nn.functional.softplus(
                -true_delta[material].sign() * pred_delta[material]
                / float(critic_config["ranking_temperature"])
            ).mean() if torch.any(material) else prediction.sum() * 0.0
            loss = value_loss + float(critic_config["ranking_weight"]) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % int(critic_config["selection_stride"]) == 0 or epoch == epochs:
            prediction_z = critic_predict(model, inputs, data, selection, device)
            prediction_log = prediction_z * target_std + target_mean
            metrics = state_metrics(prediction_log, data, selection)
            recovery = landscape_gain_recovery(prediction_log, data, selection)
            score = critic_selection_score(metrics, recovery)
            history.append({
                "epoch": epoch, "fit_loss": float(np.mean(losses)),
                "selection_score": score, "selection_metrics": metrics,
                "selection_landscape_gain_recovery": recovery,
            })
            if score < best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise AssertionError("Critic selection failed")
    model.load_state_dict(best_state, strict=True)
    return model, {
        "best_epoch": best_epoch, "best_selection_score": best_score,
        "selection_history": history, "target_mean": target_mean, "target_std": target_std,
        "target_transform": "state-balanced fit-only standardized log1p Query J50",
    }


def no_anchor_invariance(actor: torch.nn.Module, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
                         device: torch.device) -> dict[str, float]:
    local = rows[:min(20, len(rows))]
    tensors = [torch.from_numpy(value[local]).to(device) for value in inputs]
    actor.eval()
    with torch.no_grad():
        reference = actor(*tensors)[1]
        errors = {}
        for name, index in (("anchor", 3), ("feedback", 4), ("gradient", 5)):
            changed = list(tensors)
            changed[index] = torch.randn_like(changed[index])
            errors[name] = float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu())
    return errors


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["outputs"]["pretrain"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    if config["formal_validation_or_test_consumed"] or config.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("sealed-data contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    data, source_manifest, collection_manifest = load_data(config)
    device = torch.device(args.device)
    query_model = QueryDeploymentModel.from_checkpoint(Path(source_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device),
    )
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    seeds = [int(value) for value in config["critic"]["seeds"]]
    count, maximum = data["candidate_cost"].shape
    oof_actor_knots = np.empty((len(seeds), count, 8, 2), np.float32)
    oof_actor_cost = np.empty((len(seeds), count), np.float32)
    oof_twin_log = np.full((len(seeds), count, maximum), np.nan, np.float32)
    records = []
    for fold in range(5):
        oof = np.flatnonzero(data["fold_id"] == fold)
        selection_fold = (fold + 1) % 5
        selection = np.flatnonzero(data["fold_id"] == selection_fold)
        fit = np.flatnonzero((data["fold_id"] != fold) & (data["fold_id"] != selection_fold))
        if (len(fit), len(selection), len(oof)) != (360, 120, 120):
            raise AssertionError("unexpected nested split")
        episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
        if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
            raise AssertionError("episode leakage")
        inputs, normalizer = normalized_inputs(data, fit)
        for seed_index, seed in enumerate(seeds):
            run_seed = 10_000 + 100 * fold + seed
            print(f"fold={fold} seed={seed}: Actor", flush=True)
            actor, actor_training = train_actor(data, inputs, fit, selection, run_seed, config, controller, device)
            actor_splits = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                prediction = actor_predict(actor, inputs, rows, device)
                cost = query_cost(controller, data, prediction, rows)
                actor_splits[name] = actor_metrics(
                    cost, data["warm_cost"][rows], data["actor_target_cost"][rows]
                )
                if name == "oof":
                    oof_actor_knots[seed_index, rows] = prediction
                    oof_actor_cost[seed_index, rows] = cost
            critics, trainings, critic_splits = [], [], []
            for twin in range(2):
                critic_seed = 20_000 + fold * 100 + seed * 10 + twin
                print(f"fold={fold} seed={seed}: Critic {twin + 1}", flush=True)
                critic, training = train_critic(data, inputs, fit, selection, critic_seed, config, device)
                splits = {}
                for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                    prediction = critic_predict(critic, inputs, data, rows, device)
                    physical = prediction * training["target_std"] + training["target_mean"]
                    splits[name] = {
                        "metrics": state_metrics(physical, data, rows),
                        "landscape_gain_recovery": landscape_gain_recovery(physical, data, rows),
                    }
                critics.append(critic)
                trainings.append(training)
                critic_splits.append(splits)
            twin_splits = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                physical = []
                for critic, training in zip(critics, trainings):
                    prediction = critic_predict(critic, inputs, data, rows, device)
                    physical.append(prediction * training["target_std"] + training["target_mean"])
                conservative = np.fmax(physical[0], physical[1])
                twin_splits[name] = {
                    "metrics": state_metrics(conservative, data, rows),
                    "landscape_gain_recovery": landscape_gain_recovery(conservative, data, rows),
                }
                if name == "oof":
                    oof_twin_log[seed_index, rows] = conservative
            checkpoint = checkpoint_dir / f"pretrain_fold{fold}_seed{seed}.pt"
            invariance = no_anchor_invariance(actor, inputs, oof, device)
            torch.save(
                {
                    "qualification": "QUERY_SINGLE_CENTER_PRETRAIN_TRAIN_ONLY",
                    "fold": fold, "seed": seed, "selection_fold": selection_fold,
                    "actor_architecture": "DirectNoAnchorGTXActor",
                    "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    "actor_training": actor_training,
                    "critic_architecture": "ConfigurableAbsoluteActionValueCritic(base,pair_delta=false)",
                    "critic1_state_dict": {k: v.detach().cpu() for k, v in critics[0].state_dict().items()},
                    "critic2_state_dict": {k: v.detach().cpu() for k, v in critics[1].state_dict().items()},
                    "critic1_training": trainings[0], "critic2_training": trainings[1],
                    "normalization": normalizer.to_dict(), "fit_indices": fit,
                    "selection_indices": selection, "oof_indices": oof,
                    "source_replay_sha256": str(data["source_replay_sha256"]),
                    "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
                    "no_anchor_invariance": invariance,
                    "formal_validation_or_test_consumed": False,
                    "dbm_fields_or_labels_consumed": [],
                }, checkpoint,
            )
            record = {
                "fold": fold, "seed": seed, "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint), "actor": actor_splits,
                "critic1": critic_splits[0], "critic2": critic_splits[1],
                "critic_twin_conservative": twin_splits, "no_anchor_invariance": invariance,
            }
            records.append(record)
            report = twin_splits["oof"]
            print(
                f"fold={fold} seed={seed} OOF twin corr={report['metrics']['state_pearson_log_cost']['median']:.3f} "
                f"pair={report['metrics']['state_pair_sign_accuracy']['median']:.3f} "
                f"landR={report['landscape_gain_recovery']:.3f}", flush=True,
            )
    oof_path = output / "oof_predictions.npz"
    np.savez_compressed(
        oof_path, row_index=data["row_index"], fold_id=data["fold_id"],
        episode_id=data["episode_id"], seeds=np.asarray(seeds), actor_knots=oof_actor_knots,
        actor_direct_cost=oof_actor_cost, twin_conservative_log_cost=oof_twin_log,
        candidate_valid_mask=data["candidate_valid_mask"],
    )
    pooled = []
    thresholds = config["critic_pretrain_gates"]
    for seed_index, seed in enumerate(seeds):
        metrics = state_metrics(oof_twin_log[seed_index], data, np.arange(count))
        recovery = landscape_gain_recovery(oof_twin_log[seed_index], data, np.arange(count))
        offline_pass = (
            metrics["state_pearson_log_cost"]["median"] >= float(thresholds["oof_log_cost_pearson_median_minimum"])
            and metrics["state_pair_sign_accuracy"]["median"] >= float(thresholds["oof_same_state_pair_sign_accuracy_median_minimum"])
            and recovery >= float(thresholds["oof_landscape_bank_gain_recovery_median_minimum"])
        )
        pooled.append({
            "seed": seed, "metrics": metrics, "landscape_gain_recovery": recovery,
            "offline_gates_pass": bool(offline_pass),
            "actor": actor_metrics(oof_actor_cost[seed_index], data["warm_cost"], data["actor_target_cost"]),
        })
    offline_pass_count = int(sum(value["offline_gates_pass"] for value in pooled))
    qualification = (
        "QUERY_SINGLE_CENTER_PRETRAIN_OFFLINE_PASS_PENDING_FRESH_FD"
        if offline_pass_count >= int(thresholds["minimum_pooled_seeds_passing_all_gates"])
        else "QUERY_SINGLE_CENTER_PRETRAIN_OFFLINE_FAIL_NO_OAC"
    )
    summary = {
        "qualification": qualification, "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config, "pooled_oof_by_seed": pooled,
        "offline_seed_pass_count": offline_pass_count, "records": records,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-single-center-pretrain-v1",
        "dataset_type": "train-only-query-noanchor-actor-state-balanced-twin-critic",
        "qualification": qualification, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "source_replay": str(data["source_dir"]), "source_replay_sha256": str(data["source_replay_sha256"]),
        "query_checkpoint": source_manifest["query_checkpoint"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path), "oof_predictions_sha256": sha256(oof_path),
        "checkpoint_sha256": {Path(r["checkpoint"]).name: r["checkpoint_sha256"] for r in records},
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"qualification": qualification, "offline_seed_pass_count": offline_pass_count,
                      "pooled": pooled}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
