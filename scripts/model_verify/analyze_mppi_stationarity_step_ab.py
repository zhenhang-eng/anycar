#!/usr/bin/env python3
"""DBM step audit for V0/S1/S2 absolute-value Critic gradients.

The update deliberately preserves gradient magnitude:
  delta = clip(-eta * d log(1+Q)/d a_abs, +/- 0.05 * MPPI_sigma)
It never normalizes the gradient direction.  Positive gain means the frozen DBM
cost decreased.  This is a train-only mechanism audit, not an Actor update.
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
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs


ROOTS = {
    "V0": Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"),
    "S1_lambda_0.1": Path(
        "outputs/mppi_proposal/absolute_action_value_critic_stationary_20260820_v1"
    ),
    "S2_lambda_1.0": Path(
        "outputs/mppi_proposal/absolute_action_value_critic_stationary_strong_20260820_v1"
    ),
}
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-v1", type=Path,
        default=Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--etas", default="0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--cap-sigma", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def summary(value: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(value)), "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "p90": float(np.quantile(value, 0.90)),
        "worst": float(np.min(value)),
        "positive_fraction": float(np.mean(value > 1e-6)),
        "nonnegative_fraction": float(np.mean(value >= -1e-6)),
    }


def evaluate_cost(backend, weights, params, knots, states, current, reference,
                  batch_size, device):
    values = []
    for start in range(0, len(knots), batch_size):
        stop = min(start + batch_size, len(knots))
        action = interpolate_knots(
            torch.from_numpy(knots[start:stop]).to(device), params.horizon
        ).unsqueeze(1)
        with torch.no_grad():
            cost = batched_cost(
                backend, weights, action,
                torch.from_numpy(states[start:stop]).to(device),
                torch.from_numpy(current[start:stop]).to(device),
                torch.from_numpy(reference[start:stop]).to(device),
            )[:, 0]
        values.append(cost.cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(ROOTS["V0"] / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current, reference, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    sigma = np.asarray(params.noise_sigma, np.float32).reshape(1, 1, 2)
    cap = args.cap_sigma * sigma
    etas = [float(value) for value in args.etas.split(",")]
    anchors_to_audit = {"warm": 0, "bank_best": 2}
    result_rows = []
    arrays = {}

    for arm, root in ROOTS.items():
        with np.load(root / "gradient_audit.npz", allow_pickle=False) as audit:
            anchors = np.asarray(audit["anchors"], np.float32)
            seed_gradients = {
                int(key.rsplit("_", 1)[1]): np.asarray(audit[key], np.float32)
                for key in audit.files if key.startswith("predicted_seed_")
            }
        for seed, gradient in sorted(seed_gradients.items()):
            arrays[f"{arm}_seed{seed}_gradient"] = gradient
            for anchor_name, anchor_index in anchors_to_audit.items():
                base_knots = anchors[:, anchor_index]
                base_cost = evaluate_cost(
                    backend, weights, params, base_knots, states, current, reference,
                    args.batch_size, args.device,
                )
                for eta in etas:
                    requested = -eta * gradient[:, anchor_index]
                    bounded = np.clip(requested, -cap, cap)
                    candidate = np.clip(base_knots + bounded, -1.0, 1.0).astype(np.float32)
                    effective = candidate - base_knots
                    cost = evaluate_cost(
                        backend, weights, params, candidate, states, current, reference,
                        args.batch_size, args.device,
                    )
                    gain = base_cost - cost
                    normalized_rms = np.sqrt(np.mean((effective / sigma) ** 2, axis=(1, 2)))
                    record = {
                        "arm": arm, "seed": seed, "anchor": anchor_name, "eta": eta,
                        "cap_sigma": args.cap_sigma,
                        "gain": summary(gain),
                        "base_cost_mean": float(np.mean(base_cost)),
                        "candidate_cost_mean": float(np.mean(cost)),
                        "normalized_step_rms": summary(normalized_rms),
                        "clipped_component_fraction": float(
                            np.mean(np.abs(requested) > cap + 1e-12)
                        ),
                    }
                    result_rows.append(record)
                    key = f"{arm}_seed{seed}_{anchor_name}_eta_{eta:g}".replace(".", "p")
                    arrays[f"{key}_gain"] = gain
                    arrays[f"{key}_step"] = effective
                    print(
                        arm, seed, anchor_name, eta,
                        f"gain mean={record['gain']['mean']:.4f}",
                        f"p05={record['gain']['p05']:.4f}",
                        f"worst={record['gain']['worst']:.4f}",
                        f"step={record['normalized_step_rms']['median']:.4f}sigma",
                        flush=True,
                    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "TRAIN_ONLY_STATIONARITY_LIMITED_STEP_MECHANISM_AUDIT",
        "contract": {
            "seeds": "all available in each arm's gradient_audit.npz",
            "update": "-eta * predicted absolute-action log-cost gradient",
            "normalization": "none", "per_component_cap_sigma": args.cap_sigma,
            "etas": etas, "anchors": list(anchors_to_audit),
            "formal_validation_test": "not read; sealed",
        },
        "rows": result_rows,
    }
    np.savez_compressed(args.output_dir / "per_state.npz", **arrays)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
