#!/usr/bin/env python3
"""Phase 2 A0: bounded-residual Actor distillation with episode-grouped
cross-fit (the actor-learnability measurement).

Contract (review 11.46/11.47):
- Labels: the Phase 1b guarded multi-128 teachers; stay states carry a zero
  residual by construction.
- Inputs are semantic-clean physical state only: initial_state_six, raw
  reference[1:] (50x4), current action, and the explicit anchor a0.
  IQR standardization is fitted on train-fold states only.
- Output: Delta_a = 2 sigma per-dimension times tanh(f_theta(s, a0)); the
  normalized residual target therefore lives in [-2, 2] sigma. An auxiliary
  stay logit is trained with a small BCE weight but never gates the output.
- Episode-grouped 5-fold cross-fit stratified by speed, 3 seeds; inside each
  training portion a 90/10 episode split drives early stopping.
- Evaluation is real DBM rollout of the predicted centers, never MSE:
  report teacher / train-actor / heldout-actor R_a0 recovery, the
  distillation and generalization gaps, stay recall/precision, regression
  tail (P05/worst of j_pred - j_a0), and speed strata.
- Pre-registered interpretation branches (measurement, not pass/fail):
  heldout recovery >= 0.5 x teacher -> distillation works, proceed to
  actor-visited re-search; train high but heldout ~ 0 -> cross-episode
  generalization wall; train also low -> distillation/optimization problem.
- Also stores the B0 per-state gradient norm ||grad J(a0)|| as a sensitivity
  scale for a possible A1 influence normalization (fold-train-only use).

The deployment actor stays frozen; formal validation/test stay sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)
from analyze_mppi_proximal_mechanism_join import term_gradients

DEFAULT_RUN = Path("outputs/mppi_proposal/proximal_search_phase1b_20260818_v1")
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/phase2a0_residual_actor_20260818_v1")
FOLDS = 5
SEEDS = (0, 1, 2)
MAX_EPOCHS = 300
PATIENCE = 30
DELTA_MAX_SIGMA = 2.0
STAY_BCE_WEIGHT = 0.1
HIDDEN = 512
MAIN3 = ("position", "yaw", "vx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def robust_scale(block: np.ndarray) -> np.ndarray:
    quarter = np.quantile(block, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    return np.where(scale > 1e-9, scale, np.std(block, axis=0) + 1e-9)


def build_model(input_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, HIDDEN),
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, 17),
    )


def evaluate_centers(
    backend, weights, params, states, device, centers: np.ndarray
) -> np.ndarray:
    knots = torch.as_tensor(
        centers.reshape(-1, 8, 2)[:, None], dtype=torch.float32, device=device
    )
    actions = interpolate_knots(knots, params.horizon)
    with torch.no_grad():
        return batched_cost(
            backend, weights, actions,
            torch.as_tensor(
                np.stack([s["initial_state_six"] for s in states]),
                dtype=torch.float32, device=device,
            ),
            torch.as_tensor(
                np.stack([s["current_action"] for s in states]),
                dtype=torch.float32, device=device,
            ),
            torch.as_tensor(
                np.stack([s["reference"] for s in states]),
                dtype=torch.float32, device=device,
            ),
        ).squeeze(-1).cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = np.load(args.run / "labels.npz", allow_pickle=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
    keys = [f"{row['episode']}#{row['snapshot']}" for row in manifest["states"]]
    if [str(value) for value in labels["episodes"]] != keys:
        raise AssertionError("labels/manifest order mismatch")

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {
        f"{s['episode']}#{s['snapshot']}": s for s in load_states(loader_args)
    }
    states = [by_key[key] for key in keys]
    count = len(states)
    device = torch.device(args.device)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    low = np.tile(np.asarray(params.action_min, np.float32), 8)
    high = np.tile(np.asarray(params.action_max, np.float32), 8)

    features = np.concatenate([
        np.stack([s["initial_state_six"] for s in states]),
        np.stack([s["reference"].reshape(-1) for s in states]),
        np.stack([s["current_action"] for s in states]),
        np.stack([s["a0"].reshape(-1) for s in states]),
    ], axis=1).astype(np.float32)
    sigma_tiled = np.stack([
        np.repeat(s["sigma"], 8) for s in states
    ]).astype(np.float32)
    a0_flat = np.stack([s["a0"].reshape(-1) for s in states]).astype(np.float32)
    target = (labels["label_knots"].reshape(count, -1) - a0_flat) / sigma_tiled
    stay = labels["stay"].astype(np.float32)
    j_a0 = labels["j_a0"].astype(np.float64)
    j_teacher = labels["j_teacher"].astype(np.float64)
    j16 = labels["j16"].astype(np.float64)
    episodes = np.asarray([s["episode"] for s in states])
    speeds = np.asarray([s["speed"] for s in states])
    unique_episodes, episode_index = np.unique(episodes, return_inverse=True)

    # Deterministic speed-stratified episode folds.
    episode_speed = {}
    for position in range(count):
        episode_speed.setdefault(int(episode_index[position]), []).append(
            speeds[position]
        )
    episode_order = sorted(
        range(len(unique_episodes)),
        key=lambda index: (float(np.mean(episode_speed[index])), index),
    )
    fold_of_episode = {
        index: position % args.folds
        for position, index in enumerate(episode_order)
    }
    fold_of_state = np.asarray(
        [fold_of_episode[int(episode_index[position])] for position in range(count)]
    )

    denominator_all = float(np.sum(j_a0 - j16))
    r_teacher = float(np.sum(j_a0 - j_teacher) / denominator_all)

    # B0 sensitivity scales for a possible A1 (diagnostic artifact).
    b0_norms = np.zeros(count, np.float32)
    for position, state in enumerate(states):
        grads = term_gradients(backend, weights, state, state["a0"], device)
        net = -sum(grads.values())
        b0_norms[position] = float(np.linalg.norm(net))
        if position % 150 == 0:
            print(f"[b0 {position:03d}/{count:03d}]", flush=True)

    fold_reports = []
    for fold in range(args.folds):
        test_mask = fold_of_state == fold
        train_pool = np.flatnonzero(~test_mask)
        train_episodes = sorted(set(int(episode_index[i]) for i in train_pool))
        for seed in SEEDS:
            rng = np.random.default_rng(9000 + 100 * fold + seed)
            shuffled = list(train_episodes)
            rng.shuffle(shuffled)
            inner_val_episodes = set(
                shuffled[: max(1, len(shuffled) // 10)]
            )
            inner_val_mask = np.asarray([
                int(episode_index[i]) in inner_val_episodes for i in train_pool
            ])
            fit_rows = train_pool[~inner_val_mask]
            val_rows = train_pool[inner_val_mask]

            scale = robust_scale(features[fit_rows]).astype(np.float32)
            scaled_fit = (features[fit_rows] / scale).astype(np.float32)
            scaled_val = (features[val_rows] / scale).astype(np.float32)
            scaled_test = (features[test_mask] / scale).astype(np.float32)
            scaled_train_all = (features[train_pool] / scale).astype(np.float32)

            torch.manual_seed(31000 + 100 * fold + seed)
            model = build_model(features.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            tensor_fit = torch.as_tensor(scaled_fit, device=device)
            target_fit = torch.as_tensor(target[fit_rows], device=device)
            stay_fit = torch.as_tensor(stay[fit_rows], device=device)
            tensor_val = torch.as_tensor(scaled_val, device=device)
            target_val = torch.as_tensor(target[val_rows], device=device)
            stay_val = torch.as_tensor(stay[val_rows], device=device)

            best_state, best_loss, patience_left = None, float("inf"), PATIENCE
            for epoch in range(MAX_EPOCHS):
                model.train()
                permutation = torch.randperm(len(tensor_fit), device=device)
                for start in range(0, len(tensor_fit), 64):
                    rows = permutation[start:start + 64]
                    prediction = model(tensor_fit[rows])
                    residual = 2.0 * torch.tanh(prediction[:, :16])
                    loss = torch.mean((residual - target_fit[rows]) ** 2)
                    loss = loss + STAY_BCE_WEIGHT * torch.nn.functional.binary_cross_entropy_with_logits(
                        prediction[:, 16], stay_fit[rows]
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    prediction = model(tensor_val)
                    residual = 2.0 * torch.tanh(prediction[:, :16])
                    val_loss = float(torch.mean(
                        (residual - target_val) ** 2
                    ))
                if val_loss < best_loss - 1e-6:
                    best_loss, patience_left = val_loss, PATIENCE
                    best_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            def predict(rows: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    out = model(torch.as_tensor(rows, device=device))
                    delta = 2.0 * torch.tanh(out[:, :16])
                    stay_logit = out[:, 16]
                return delta.cpu().numpy(), stay_logit.cpu().numpy()

            def rollout_gain(rows: np.ndarray, scaled: np.ndarray):
                delta, _ = predict(scaled)
                centers = np.clip(
                    a0_flat[rows] + delta * sigma_tiled[rows], low, high
                )
                costs = evaluate_centers(
                    backend, weights, params,
                    [states[i] for i in rows], device, centers,
                ).astype(np.float64)
                return costs

            costs_train = rollout_gain(train_pool, scaled_train_all)
            costs_test = rollout_gain(
                np.flatnonzero(test_mask), scaled_test
            )
            gain_train = j_a0[train_pool] - costs_train
            gain_test = j_a0[test_mask] - costs_test
            r_train = float(np.sum(gain_train) / np.sum(j_a0[train_pool] - j16[train_pool]))
            r_test = float(np.sum(gain_test) / np.sum(j_a0[test_mask] - j16[test_mask]))

            delta_test, logit_test = predict(scaled_test)
            pred_norm = np.sqrt(np.mean(delta_test ** 2, axis=1))
            true_stay = stay[test_mask] > 0.5
            stay_recall = float(np.mean(
                pred_norm[true_stay] < 0.05
            )) if np.any(true_stay) else None
            stay_precision = float(np.mean(
                stay[test_mask][pred_norm < 0.05]
            )) if np.any(pred_norm < 0.05) else None
            regression = costs_test - j_a0[test_mask]
            fold_reports.append({
                "fold": fold, "seed": seed,
                "best_epoch": epoch, "inner_val_loss": best_loss,
                "r_train": r_train, "r_heldout": r_test,
                "heldout_gain_median": float(np.median(gain_test)),
                "heldout_gain_p05": float(np.quantile(gain_test, 0.05)),
                "heldout_gain_worst": float(np.min(gain_test)),
                "heldout_regression_fraction": float(np.mean(regression > 1e-6)),
                "stay_recall": stay_recall,
                "stay_precision": stay_precision,
                "mover_count": int(np.sum(~true_stay)),
                "stay_count": int(np.sum(true_stay)),
            })
            print(json.dumps(fold_reports[-1]), flush=True)

    heldout_r = np.asarray([row["r_heldout"] for row in fold_reports])
    train_r = np.asarray([row["r_train"] for row in fold_reports])
    worst = np.asarray([row["heldout_gain_worst"] for row in fold_reports])
    p05 = np.asarray([row["heldout_gain_p05"] for row in fold_reports])
    recalls = [row["stay_recall"] for row in fold_reports if row["stay_recall"] is not None]

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PHASE2A0_ACTOR_LEARNABILITY_MEASURED_ACTOR_STILL_FROZEN",
        "sources": {
            "phase1b_run": str(args.run),
            "phase1b_labels_sha256": sha256_file(args.run / "labels.npz"),
            "repeat": args.repeat,
        },
        "protocol": {
            "states": count,
            "folds": args.folds,
            "seeds": list(SEEDS),
            "input": "six + reference[1:] + current_action + a0, fold-train IQR",
            "output": "Delta_a = 2 sigma * tanh(f), auxiliary stay logit",
            "stay_bce_weight": STAY_BCE_WEIGHT,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "interpretation_branches": (
                "heldout >= 0.5 x teacher -> proceed; train high heldout ~ 0 "
                "-> generalization wall; train low -> distillation problem"
            ),
        },
        "results": {
            "r_teacher": r_teacher,
            "r_train_median": float(np.median(train_r)),
            "r_heldout_median": float(np.median(heldout_r)),
            "r_heldout_min": float(np.min(heldout_r)),
            "r_heldout_max": float(np.max(heldout_r)),
            "heldout_gain_p05_median": float(np.median(p05)),
            "heldout_gain_worst_min": float(np.min(worst)),
            "stay_recall_median": float(np.median(recalls)) if recalls else None,
        },
        "folds": fold_reports,
        "b0_note": (
            "per-state ||grad J(a0)|| stored for a possible A1 influence "
            "normalization; fold-train-only usage; never an online input"
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.output / "b0_sensitivity.npz",
        episodes=labels["episodes"],
        gradient_norm=b0_norms,
    )
    print(json.dumps(summary["results"], indent=1))


if __name__ == "__main__":
    main()
