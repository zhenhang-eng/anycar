#!/usr/bin/env python3
"""Compare original and absorbed Critic Actor directions at matched output RMS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import actor_predict, distribution, load_data, sha256
from train_highspeed_actor_visited_oac import SIGMA, build_inputs, rollout_bank


DEFAULT_SOURCE = Path("outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1")
DEFAULT_BUDGET = Path("outputs/mppi_proposal/highspeed_search_replay_critic_budget_20260830_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--budget-dir", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-radii", default="0.005,0.01,0.02")
    parser.add_argument("--raw-cost-weight-cap", type=float, default=8.0)
    parser.add_argument("--actor-lr-reference", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_data(source_root: Path) -> dict[str, np.ndarray]:
    contract = json.loads((source_root / "contract.json").read_text())
    pretrain_root = Path(contract["source_pretrain"])
    summary = json.loads((pretrain_root / "summary.json").read_text())
    source = summary["source"]
    data = load_data(Path(source["replay_dir"]), Path(source["teacher_dir"]))
    indices = np.asarray(summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    return {
        key: value[indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }


def load_actor(payload: dict, device: torch.device) -> DirectNoAnchorGTXActor:
    actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    actor.load_state_dict(payload["actor_selected_state_dict"], strict=True)
    return actor


def load_critics(payload: dict, device: torch.device):
    critics = []
    training = []
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        critics.append(critic)
        training.append(payload[f"critic{twin}_training"])
    return tuple(critics), tuple(training)


def actor_gradient(
    actor: DirectNoAnchorGTXActor, critics, training, inputs,
    rows: np.ndarray, cap: float, device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    actor.eval()
    tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
    center = actor(*tensors)[1]
    physical = []
    for critic, spec in zip(critics, training):
        prediction = critic(tensors[0], tensors[1], tensors[2], center[:, None])[:, 0]
        physical.append(
            prediction * float(spec["target_std"]) + float(spec["target_mean"])
        )
    conservative = torch.maximum(physical[0], physical[1])
    weight = torch.exp(conservative.detach() - conservative.detach().mean()).clamp(max=cap)
    weight = weight / weight.mean().clamp_min(1e-12)
    objective = (weight * conservative).mean()
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    gradient_raw = torch.autograd.grad(objective, parameters)
    gradient = [value.detach().cpu() for value in gradient_raw]
    norm = float(np.sqrt(sum(float(torch.sum(value.double() ** 2)) for value in gradient)))
    ess = float((weight.sum().square() / (weight.square().sum() * len(weight))).detach())
    return gradient, {
        "objective": float(objective.detach()), "parameter_gradient_norm": norm,
        "raw_weight_ess_fraction": ess, "raw_weight_max": float(weight.max().detach()),
    }


def gradient_cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    dot = sum(float(torch.sum(a.double() * b.double())) for a, b in zip(left, right))
    left_norm = np.sqrt(sum(float(torch.sum(a.double() ** 2)) for a in left))
    right_norm = np.sqrt(sum(float(torch.sum(b.double() ** 2)) for b in right))
    return float(dot / max(left_norm * right_norm, 1e-30))


def output_rms(action: np.ndarray, base: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((action - base) / SIGMA.reshape(1, 1, 2)) ** 2)))


def raw_sgd_step(
    actor: DirectNoAnchorGTXActor, gradient: list[torch.Tensor], inputs,
    fit: np.ndarray, base_fit: np.ndarray, target: float, device: torch.device,
) -> tuple[DirectNoAnchorGTXActor, float, float]:
    scale = 1e-6
    candidate = None
    actual = 0.0
    for _ in range(12):
        candidate = copy.deepcopy(actor).to(device)
        with torch.no_grad():
            for parameter, value in zip(candidate.parameters(), gradient):
                parameter.add_(value.to(device), alpha=-scale)
        action = actor_predict(candidate, inputs, fit, device)
        actual = output_rms(action, base_fit)
        if actual <= 1e-14:
            scale *= 10.0
            continue
        ratio = target / actual
        if abs(ratio - 1.0) <= 0.005:
            break
        scale *= float(np.clip(ratio, 0.1, 10.0))
    assert candidate is not None
    return candidate, actual, float(scale)


def adam_step(
    actor: DirectNoAnchorGTXActor, gradient: list[torch.Tensor], optimizer_state: dict,
    inputs, fit: np.ndarray, base_fit: np.ndarray, target: float,
    reference_lr: float, weight_decay: float, device: torch.device,
) -> tuple[DirectNoAnchorGTXActor, float, float]:
    learning_rate = reference_lr
    candidate = None
    actual = 0.0
    for _ in range(12):
        candidate = copy.deepcopy(actor).to(device)
        optimizer = torch.optim.AdamW(
            candidate.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay
        optimizer.zero_grad(set_to_none=True)
        for parameter, value in zip(candidate.parameters(), gradient):
            parameter.grad = value.to(device).clone()
        optimizer.step()
        action = actor_predict(candidate, inputs, fit, device)
        actual = output_rms(action, base_fit)
        if actual <= 1e-14:
            learning_rate *= 10.0
            continue
        ratio = target / actual
        if abs(ratio - 1.0) <= 0.005:
            break
        learning_rate *= float(np.clip(ratio, 0.1, 10.0))
    assert candidate is not None
    return candidate, actual, float(learning_rate)


def rollout_actions(data, actions, rows, backend, weights, params, device, batch_size):
    return rollout_bank(
        data, actions[:, None], rows, backend, weights, params, device, batch_size
    )[:, 0]


def evaluation(base: np.ndarray, cost: np.ndarray, speed: np.ndarray) -> dict[str, Any]:
    gain = np.asarray(base, np.float64) - np.asarray(cost, np.float64)
    result = {
        "cost": distribution(cost), "gain": distribution(gain),
        "regression_fraction": float(np.mean(gain < -1e-5)),
        "by_speed": {},
    }
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result["by_speed"][f"{float(value):.0f}"] = {
            "count": int(mask.sum()), "gain_mean": float(gain[mask].mean()),
            "gain_median": float(np.median(gain[mask])),
            "gain_p05": float(np.quantile(gain[mask], 0.05)),
            "regression_fraction": float(np.mean(gain[mask] < -1e-5)),
        }
    return result


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.resolve()
    budget_root = args.budget_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    radii = [float(value) for value in args.target_radii.split(",") if value.strip()]
    source_summary_path = source_root / "summary.json"
    source_validator_path = source_root / "validator_report.json"
    budget_summary_path = budget_root / "summary.json"
    budget_validator_path = budget_root / "validator_report.json"
    source_summary = json.loads(source_summary_path.read_text())
    budget_summary = json.loads(budget_summary_path.read_text())
    if json.loads(source_validator_path.read_text())["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("source validation missing")
    if json.loads(budget_validator_path.read_text())["qualification"] != "HIGHSPEED_SEARCH_REPLAY_CRITIC_BUDGET_INDEPENDENT_RELOAD_PASS":
        raise AssertionError("budget validation missing")
    source_records = {
        (int(row["fold"]), int(row["seed"])): row for row in source_summary["records"]
    }
    budget_records = {
        (int(row["fold"]), int(row["seed"])): row for row in budget_summary["records"]
    }
    data = load_shared_data(source_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    records = []
    saved = {
        "fold": [], "seed": [], "arm": [], "method": [], "radius": [],
        "oof": [], "action": [], "cost": [], "base_action": [], "base_cost": [],
    }
    for key in sorted(source_records):
        fold, seed = key
        source_record = source_records[key]
        budget_record = budget_records[key]
        source_path = Path(source_record["checkpoint"])
        budget_path = Path(budget_record["checkpoint"])
        if sha256(source_path) != source_record["checkpoint_sha256"]:
            raise AssertionError("source hash mismatch")
        if sha256(budget_path) != budget_record["checkpoint_sha256"]:
            raise AssertionError("budget hash mismatch")
        source = torch.load(source_path, map_location=device)
        budget = torch.load(budget_path, map_location=device)
        fit = np.asarray(source["fit_indices"], np.int64)
        selection = np.asarray(source["selection_indices"], np.int64)
        oof = np.asarray(source["oof_indices"], np.int64)
        if np.intersect1d(data["episode"][fit], data["episode"][oof]).size:
            raise AssertionError("episode leakage")
        inputs = build_inputs(data, source)
        actor = load_actor(source, device)
        base_actions = {
            name: actor_predict(actor, inputs, rows, device)
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof))
        }
        base_costs = {
            name: rollout_actions(
                data, base_actions[name], rows, backend, weights, params, device,
                args.rollout_batch_size,
            )
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof))
        }
        gradients = {}
        gradient_info = {}
        for arm, payload in (("original", source), ("absorbed1600", budget)):
            critics, training = load_critics(payload, device)
            gradients[arm], gradient_info[arm] = actor_gradient(
                actor, critics, training, inputs, fit,
                args.raw_cost_weight_cap, device,
            )
        run = {
            "fold": fold, "seed": seed,
            "source_checkpoint_sha256": source_record["checkpoint_sha256"],
            "budget_checkpoint_sha256": budget_record["checkpoint_sha256"],
            "gradient": gradient_info,
            "original_absorbed_parameter_gradient_cosine": gradient_cosine(
                gradients["original"], gradients["absorbed1600"]
            ),
            "steps": [],
        }
        for arm in ("original", "absorbed1600"):
            for method in ("raw_sgd", "saved_adam"):
                for radius in radii:
                    if method == "raw_sgd":
                        candidate, actual, scale = raw_sgd_step(
                            actor, gradients[arm], inputs, fit, base_actions["fit"],
                            radius, device,
                        )
                    else:
                        candidate, actual, scale = adam_step(
                            actor, gradients[arm], source["actor_optimizer"], inputs,
                            fit, base_actions["fit"], radius,
                            args.actor_lr_reference, args.weight_decay, device,
                        )
                    split_metrics = {}
                    oof_action = None
                    oof_cost = None
                    for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                        action = actor_predict(candidate, inputs, rows, device)
                        cost = rollout_actions(
                            data, action, rows, backend, weights, params, device,
                            args.rollout_batch_size,
                        )
                        split_metrics[name] = evaluation(
                            base_costs[name], cost, data["speed"][rows]
                        )
                        if name == "oof":
                            oof_action, oof_cost = action, cost
                    run["steps"].append({
                        "arm": arm, "method": method, "target_rms_sigma": radius,
                        "actual_fit_rms_sigma": actual, "parameter_scale": scale,
                        "metrics": split_metrics,
                    })
                    saved["fold"].append(fold); saved["seed"].append(seed)
                    saved["arm"].append(arm); saved["method"].append(method)
                    saved["radius"].append(radius); saved["oof"].append(oof)
                    saved["action"].append(oof_action); saved["cost"].append(oof_cost)
                    saved["base_action"].append(base_actions["oof"])
                    saved["base_cost"].append(base_costs["oof"])
        records.append(run)
        print(
            f"fold={fold} seed={seed} gradient_cosine="
            f"{run['original_absorbed_parameter_gradient_cosine']:.4f}", flush=True,
        )
    arrays_path = output / "evaluation.npz"
    np.savez_compressed(
        arrays_path,
        fold=np.asarray(saved["fold"], np.int64),
        seed=np.asarray(saved["seed"], np.int64),
        arm=np.asarray(saved["arm"]), method=np.asarray(saved["method"]),
        radius=np.asarray(saved["radius"], np.float32),
        oof=np.stack(saved["oof"]), action=np.stack(saved["action"]),
        cost=np.stack(saved["cost"]), base_action=np.stack(saved["base_action"]),
        base_cost=np.stack(saved["base_cost"]),
    )
    aggregate = {}
    for method in ("raw_sgd", "saved_adam"):
        aggregate[method] = {}
        for radius in radii:
            token = f"{radius:.3f}"
            aggregate[method][token] = {}
            for arm in ("original", "absorbed1600"):
                rows = [
                    step for record in records for step in record["steps"]
                    if step["method"] == method and step["arm"] == arm
                    and np.isclose(step["target_rms_sigma"], radius)
                ]
                aggregate[method][token][arm] = {
                    "actual_fit_rms_sigma": distribution([
                        row["actual_fit_rms_sigma"] for row in rows
                    ]),
                    "oof_gain_mean": distribution([
                        row["metrics"]["oof"]["gain"]["mean"] for row in rows
                    ]),
                    "oof_gain_median": distribution([
                        row["metrics"]["oof"]["gain"]["median"] for row in rows
                    ]),
                    "oof_gain_p05": distribution([
                        row["metrics"]["oof"]["gain"]["p05"] for row in rows
                    ]),
                    "oof_regression_fraction": distribution([
                        row["metrics"]["oof"]["regression_fraction"] for row in rows
                    ]),
                }
                if arm == "absorbed1600":
                    original = [
                        step for record in records for step in record["steps"]
                        if step["method"] == method and step["arm"] == "original"
                        and np.isclose(step["target_rms_sigma"], radius)
                    ]
                    aggregate[method][token]["paired_absorbed_minus_original"] = {
                        "oof_gain_mean": distribution([
                            right["metrics"]["oof"]["gain"]["mean"]
                            - left["metrics"]["oof"]["gain"]["mean"]
                            for left, right in zip(original, rows)
                        ]),
                        "oof_gain_p05": distribution([
                            right["metrics"]["oof"]["gain"]["p05"]
                            - left["metrics"]["oof"]["gain"]["p05"]
                            for left, right in zip(original, rows)
                        ]),
                    }
    analysis = {
        "format": "highspeed_matched_actor_step_critic_ab_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_MATCHED_ACTOR_OUTPUT_STEP_CRITIC_AB_COMPLETE_TRAIN_ONLY",
        "contract": {
            "gradient_source": "continuously-trained Twin Critic only",
            "gradient_states": "fit split; full-batch gamma=1 raw-J-weighted Actor objective",
            "matched_measure": "fit Actor output RMS normalized by [0.25,0.35]",
            "DBM_role": "post-step scalar J50 evaluation only; no analytic gradient",
            "formal_validation_or_test_created": False,
            "target_radii_sigma": radii,
        },
        "sources": {
            "source_summary_sha256": sha256(source_summary_path),
            "source_validator_sha256": sha256(source_validator_path),
            "budget_summary_sha256": sha256(budget_summary_path),
            "budget_validator_sha256": sha256(budget_validator_path),
            "evaluation_sha256": file_sha256(arrays_path),
        },
        "gradient_cosine": distribution([
            row["original_absorbed_parameter_gradient_cosine"] for row in records
        ]),
        "aggregate": aggregate,
        "records": records,
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: value for key, value in analysis.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
