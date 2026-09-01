#!/usr/bin/env python3
"""Independently validate a frozen-state two-center integration artifact."""

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
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import batched_cost
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_RUN = Path("outputs/mppi_proposal/two_center_integration_20260813_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--replay-contexts", type=int, default=24)
    parser.add_argument("--cost-tolerance", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_replay_index(context_count: int, requested: int) -> np.ndarray:
    count = min(max(requested, 1), context_count)
    return np.unique(np.linspace(0, context_count - 1, count).round().astype(np.int64))


def load_state(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as source:
        initial = np.asarray(source["initial_state_six"], np.float32)
        current = np.asarray(source["current_action"], np.float32)
        reference = np.asarray(source["reference"], np.float32)
        if len(reference) == 51:
            reference = reference[1:]
    return initial, current, reference


def assert_within(name: str, value: float, tolerance: float) -> None:
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f"{name}: {value} exceeds {tolerance}")


@torch.no_grad()
def replay_weighted_cost(
    arrays: np.lib.npyio.NpzFile,
    summary: dict,
    arm: str,
    context_index: np.ndarray,
    device: torch.device,
) -> float:
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**summary["dbm_params"])
    )
    weights = TorchMPPICostWeights(**summary["cost_weights"])
    initial, current, reference = zip(*(
        load_state(str(arrays["source_path"][index])) for index in context_index
    ))
    initial_t = torch.from_numpy(np.asarray(initial, np.float32)).to(device)
    current_t = torch.from_numpy(np.asarray(current, np.float32)).to(device)
    reference_t = torch.from_numpy(np.asarray(reference, np.float32)).to(device)
    maximum = 0.0
    sequence = arrays[f"{arm}_weighted_sequence"]
    stored = arrays[f"{arm}_weighted_cost"]
    for seed_index in range(sequence.shape[0]):
        actions = torch.from_numpy(sequence[seed_index, context_index]).to(device).unsqueeze(1)
        replay = batched_cost(
            backend, weights, actions, initial_t, current_t, reference_t
        )[:, 0].cpu().numpy()
        maximum = max(maximum, float(np.max(np.abs(
            replay - stored[seed_index, context_index]
        ))))
    return maximum


def main() -> None:
    args = parse_args()
    summary_path = args.run_dir / "summary.json"
    arrays_path = args.run_dir / "evaluation.npz"
    summary = json.loads(summary_path.read_text())
    if summary["candidate_contract"]["num_samples"] != 256:
        raise AssertionError("candidate budget is not 256")
    if summary["candidate_contract"]["soft_remaining_candidates"] != 254:
        raise AssertionError("soft bank does not retain 254 Gaussian candidates")
    if "INTERNAL" not in summary["qualification"]:
        raise AssertionError("artifact is not explicitly internal-only")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        seeds = len(summary["evaluation_seeds"])
        contexts = int(summary["context_count"])
        expected = (seeds, contexts, 256)
        direct_warm = np.broadcast_to(arrays["direct_warm_cost"][None], (seeds, contexts))
        direct_actor = np.broadcast_to(arrays["direct_actor_cost"][None], (seeds, contexts))
        shape_checks: dict[str, list[int]] = {}
        for arm in ("warm_gaussian", "actor_replace", "soft_two_center"):
            cost = arrays[f"{arm}_candidate_cost"]
            weight = arrays[f"{arm}_candidate_weight"]
            if cost.shape != expected or weight.shape != expected:
                raise AssertionError(f"{arm} candidate shape mismatch")
            shape_checks[arm] = list(cost.shape)

        exact_errors = {
            "warm_gaussian_candidate0_vs_direct_warm": float(np.max(np.abs(
                arrays["warm_gaussian_candidate_cost"][..., 0] - direct_warm
            ))),
            "actor_replace_candidate0_vs_direct_actor": float(np.max(np.abs(
                arrays["actor_replace_candidate_cost"][..., 0] - direct_actor
            ))),
            "soft_candidate0_vs_direct_actor": float(np.max(np.abs(
                arrays["soft_two_center_candidate_cost"][..., 0] - direct_actor
            ))),
            "soft_candidate1_vs_direct_warm": float(np.max(np.abs(
                arrays["soft_two_center_candidate_cost"][..., 1] - direct_warm
            ))),
        }
        for name, value in exact_errors.items():
            assert_within(name, value, args.cost_tolerance)

        weight_errors: dict[str, float] = {}
        for arm in ("warm_gaussian", "actor_replace", "soft_two_center"):
            cost = arrays[f"{arm}_candidate_cost"].astype(np.float64)
            logits = -(cost - cost.min(axis=-1, keepdims=True))
            logits -= logits.max(axis=-1, keepdims=True)
            expected_weight = np.exp(logits)
            expected_weight /= expected_weight.sum(axis=-1, keepdims=True)
            error = float(np.max(np.abs(
                expected_weight - arrays[f"{arm}_candidate_weight"]
            )))
            weight_errors[arm] = error
            assert_within(f"{arm} softmax", error, 2e-6)

        expected_hard = np.minimum(
            arrays["soft_two_center_weighted_cost"], direct_warm
        )
        hard_error = float(np.max(np.abs(
            expected_hard - arrays["hard_fallback_cost"]
        )))
        assert_within("hard fallback reconstruction", hard_error, 1e-7)
        floor_violation = float(np.max(arrays["hard_fallback_cost"] - direct_warm))
        if floor_violation > 1e-7:
            raise AssertionError("hard fallback violates direct-warm model floor")
        expected_258 = np.minimum(
            arrays["warm_gaussian_weighted_cost"], direct_actor
        )
        guard_258_error = float(np.max(np.abs(
            expected_258 - arrays["warm_mppi_vs_direct_actor_hard_258_cost"]
        )))
        assert_within("258 guard reconstruction", guard_258_error, 1e-7)
        guard_258_floor_violation = float(np.max(
            arrays["warm_mppi_vs_direct_actor_hard_258_cost"]
            - arrays["warm_gaussian_weighted_cost"]
        ))
        if guard_258_floor_violation > 1e-7:
            raise AssertionError("258 guard violates warm-MPPI model floor")

        replay_index = stable_replay_index(contexts, args.replay_contexts)
        replay_errors = {
            arm: replay_weighted_cost(
                arrays, summary, arm, replay_index, torch.device(args.device)
            )
            for arm in ("warm_gaussian", "actor_replace", "soft_two_center")
        }
        for name, value in replay_errors.items():
            assert_within(f"{name} weighted-output replay", value, args.cost_tolerance)

        baseline = arrays["warm_gaussian_weighted_cost"]
        summary_errors = {
            "actor_gain_mean": abs(
                float(np.mean(
                    (baseline - arrays["actor_replace_weighted_cost"]).astype(np.float64)
                ))
                - summary["summaries"]["actor_replace"]["gain_vs_warm_gaussian"]["mean"]
            ),
            "soft_gain_mean": abs(
                float(np.mean(
                    (baseline - arrays["soft_two_center_weighted_cost"]).astype(np.float64)
                ))
                - summary["summaries"]["soft_two_center"]["gain_vs_warm_gaussian"]["mean"]
            ),
            "hard_gain_mean": abs(
                float(np.mean(
                    (baseline - arrays["hard_fallback_cost"]).astype(np.float64)
                ))
                - summary["summaries"]["hard_fallback"]["gain_vs_warm_gaussian"]["mean"]
            ),
            "guard_258_gain_mean": abs(
                float(np.mean(
                    (
                        baseline
                        - arrays["warm_mppi_vs_direct_actor_hard_258_cost"]
                    ).astype(np.float64)
                ))
                - summary["summaries"]["warm_mppi_vs_direct_actor_hard_258"]
                ["gain_vs_warm_gaussian"]["mean"]
            ),
        }
        for name, value in summary_errors.items():
            assert_within(name, value, 1e-6)

    validation = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "VALIDATED_INTERNAL_MECHANISM_ARTIFACT",
        "summary_sha256": sha256_file(summary_path),
        "evaluation_sha256": sha256_file(arrays_path),
        "candidate_shapes": shape_checks,
        "exact_candidate_cost_max_abs_error": exact_errors,
        "softmax_weight_max_abs_error": weight_errors,
        "hard_fallback_reconstruction_max_abs_error": hard_error,
        "hard_direct_warm_floor_max_violation": floor_violation,
        "guard_258_reconstruction_max_abs_error": guard_258_error,
        "guard_258_warm_mppi_floor_max_violation": guard_258_floor_violation,
        "weighted_output_replay_contexts": replay_index.tolist(),
        "weighted_output_replay_max_abs_error": replay_errors,
        "summary_mean_max_abs_error": summary_errors,
        "cost_tolerance": args.cost_tolerance,
        "result": "PASS",
    }
    (args.run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
