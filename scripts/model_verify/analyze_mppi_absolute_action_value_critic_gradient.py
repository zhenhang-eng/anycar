#!/usr/bin/env python3
"""Fresh action-gradient audit for the absolute-action value Critic.

The Critic was selected only by scalar value/ranking metrics.  This script
therefore independently compares its autograd derivative of log(1+J) with the
frozen DBM derivative in the same absolute 8x2 knot coordinates.  It audits
three action regimes without retraining: warm, a fixed heterogeneous raw start,
and the best candidate in the 24-plan bank.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    DEFAULT_OUTPUT,
    make_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gt-v1", type=Path,
        default=Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def distribution(value: np.ndarray) -> dict:
    return {
        "count": int(len(value)), "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
    }


def load_rollout_inputs(data: dict, gt_v1: Path):
    rows = json.loads((gt_v1 / "summary.json").read_text())["rows"]
    by_key = {(r["episode"], r["snapshot"]): r for r in rows}
    states, current, reference = [], [], []
    params_json = weights_json = dbm_json = None
    for episode, snapshot in zip(data["episode"], data["snapshot"]):
        row = by_key[(str(episode), str(snapshot))]
        with np.load(Path(row["source"]), allow_pickle=False) as source:
            states.append(np.asarray(source["initial_state_six"], np.float32))
            current.append(np.asarray(source["current_action"], np.float32))
            ref = np.asarray(source["reference"], np.float32)
            params_json = str(source["mppi_params_json"])
            weights_json = str(source["cost_weights_json"])
            dbm_json = str(source["dbm_params_json"])
            horizon = int(json.loads(params_json)["horizon"])
            if len(ref) == horizon + 1:
                ref = ref[1:]
            reference.append(ref)
    return (
        np.asarray(states, np.float32), np.asarray(current, np.float32),
        np.asarray(reference, np.float32), params_json, weights_json, dbm_json,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with np.load(args.critic_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current_action, direct_reference, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    folds = make_folds(data, args.folds)
    best_index = np.argmin(data["costs"], axis=1)
    anchors = np.stack((
        data["actions"][:, 0],
        data["actions"][:, 5],
        data["actions"][np.arange(len(best_index)), best_index],
    ), axis=1).astype(np.float32)
    anchor_names = ("warm", "raw_random_0", "bank_best")

    # True derivatives are shared across Critic seeds and computed once.
    true_gradient = np.empty_like(anchors)
    replay_cost = np.empty(anchors.shape[:2], np.float32)
    for start in range(0, len(anchors), args.batch_size):
        stop = min(start + args.batch_size, len(anchors))
        knots = torch.from_numpy(anchors[start:stop]).to(device).requires_grad_(True)
        actions = interpolate_knots(knots, params.horizon)
        cost = batched_cost(
            backend, weights, actions,
            torch.from_numpy(states[start:stop]).to(device),
            torch.from_numpy(current_action[start:stop]).to(device),
            torch.from_numpy(direct_reference[start:stop]).to(device),
        )
        gradient = torch.autograd.grad(torch.log1p(cost).sum(), knots)[0]
        true_gradient[start:stop] = gradient.detach().cpu().numpy()
        replay_cost[start:stop] = cost.detach().cpu().numpy()
    cost_contract_error = max(
        float(np.max(np.abs(replay_cost[:, 0] - data["costs"][:, 0]))),
        float(np.max(np.abs(replay_cost[:, 1] - data["costs"][:, 5]))),
        float(np.max(np.abs(replay_cost[:, 2] - data["costs"].min(1)))),
    )
    if cost_contract_error > 1e-3:
        raise AssertionError(f"DBM replay contract mismatch: {cost_contract_error}")

    seed_records = []
    prediction_artifact = {"true_gradient": true_gradient, "anchors": anchors}
    for seed in [int(x) for x in args.seeds.split(",")]:
        predicted = np.empty_like(true_gradient)
        for fold in range(args.folds):
            index = np.flatnonzero(folds == fold)
            payload = torch.load(
                args.critic_root / f"critic_seed{seed}_fold{fold}.pt",
                map_location=device,
            )
            normalizer = MPPIProposalNormalization.from_dict(
                payload["training"]["normalization"]
            )
            history, reference, current = normalizer.normalize_numpy(
                data["history"][index], data["reference"][index], data["current"][index]
            )
            model = AbsoluteActionValueCritic(dropout=0.0).to(device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            for begin in range(0, len(index), args.batch_size):
                local = slice(begin, begin + args.batch_size)
                target_index = index[local]
                action = torch.from_numpy(anchors[target_index]).to(device).requires_grad_(True)
                value = model(
                    torch.from_numpy(history[local].astype(np.float32)).to(device),
                    torch.from_numpy(reference[local].astype(np.float32)).to(device),
                    torch.from_numpy(current[local].astype(np.float32)).to(device),
                    action,
                ) * float(payload["training"]["target_std"])
                gradient = torch.autograd.grad(value.sum(), action)[0]
                predicted[target_index] = gradient.detach().cpu().numpy()
        prediction_artifact[f"predicted_seed_{seed}"] = predicted
        anchor_metrics = {}
        for anchor_index, name in enumerate(anchor_names):
            target = true_gradient[:, anchor_index].reshape(len(anchors), -1)
            estimate = predicted[:, anchor_index].reshape(len(anchors), -1)
            cosine = cosine_rows(estimate, target)
            norm_ratio = np.linalg.norm(estimate, axis=1) / np.maximum(
                np.linalg.norm(target, axis=1), 1e-12
            )
            anchor_metrics[name] = {
                "cosine": distribution(cosine),
                "cosine_positive_fraction": float(np.mean(cosine > 0)),
                "cosine_above_0_5_fraction": float(np.mean(cosine >= 0.5)),
                "norm_ratio": distribution(norm_ratio),
                "by_speed": {
                    f"{speed:.1f}": {
                        "cosine": distribution(cosine[data["speed"] == speed]),
                        "norm_ratio": distribution(norm_ratio[data["speed"] == speed]),
                    } for speed in sorted(np.unique(data["speed"]))
                },
            }
        seed_records.append({"seed": seed, "anchors": anchor_metrics})

    # Old gradient gate is intentionally applied only after the independent
    # value/ranking gate: median >= .70, P10 >= 0, norm median in [.5, 2].
    gates = []
    for row in seed_records:
        per_anchor = {}
        for name in anchor_names:
            value = row["anchors"][name]
            per_anchor[name] = {
                "cosine_median_ge_0_70": value["cosine"]["median"] >= 0.70,
                "cosine_p10_ge_0": value["cosine"]["p10"] >= 0.0,
                "norm_ratio_median_in_0_5_2": (
                    0.5 <= value["norm_ratio"]["median"] <= 2.0
                ),
            }
            per_anchor[name]["passed"] = bool(all(per_anchor[name].values()))
        gates.append({"seed": row["seed"], "anchors": per_anchor})
    all_anchor_passes = sum(
        all(anchor["passed"] for anchor in row["anchors"].values()) for row in gates
    )
    qualification = (
        "ABSOLUTE_VALUE_CRITIC_GRADIENT_USABLE"
        if all_anchor_passes >= 2
        else "ABSOLUTE_VALUE_CRITIC_GRADIENT_NOT_USABLE"
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "value_ranking_parent": str(args.critic_root / "summary.json"),
        "coordinate_contract": "d log1p(J) / d absolute 8x2 action knots",
        "anchors": list(anchor_names),
        "dbm_replay_max_abs_error": cost_contract_error,
        "records": seed_records, "gates": gates,
        "passed_seed_count_all_anchors": int(all_anchor_passes),
        "interpretation_boundary": (
            "A negative result does not invalidate scalar value/ranking use; "
            "it only forbids using autograd through this Critic as the Actor update."
        ),
    }
    np.savez_compressed(args.critic_root / "gradient_audit.npz", **prediction_artifact)
    (args.critic_root / "gradient_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(qualification)
    for row in seed_records:
        print("seed", row["seed"], {
            name: {
                "cos_med": round(row["anchors"][name]["cosine"]["median"], 3),
                "cos_p10": round(row["anchors"][name]["cosine"]["p10"], 3),
                "norm_med": round(row["anchors"][name]["norm_ratio"]["median"], 3),
            } for name in anchor_names
        })


if __name__ == "__main__":
    main()
