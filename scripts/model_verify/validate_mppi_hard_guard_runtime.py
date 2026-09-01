#!/usr/bin/env python3
"""Validate the runtime hard-guard primitive on frozen DBM contexts.

The check drives ``TorchMPPIController.hard_guard_action_sequence`` directly.
It verifies that deterministic proposal scoring does not consume the MPPI RNG or
alter warm-start state, reproduces independently stored model costs, and always
selects the lower cost of the original warm-MPPI weighted output and Actor direct
sequence.  Formal validation/test episodes are never loaded.
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
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIController,
    TorchMPPIParams,
)
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_SOURCE = Path("outputs/mppi_proposal/two_center_integration_20260813_v2")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/hard_guard_runtime_validation_20260813_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", type=int, default=60)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_indices(total: int, requested: int) -> np.ndarray:
    count = min(max(1, requested), total)
    return np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    summary_path = args.source / "summary.json"
    arrays_path = args.source / "evaluation.npz"
    source_summary = json.loads(summary_path.read_text())
    if "INTERNAL" not in source_summary["qualification"]:
        raise AssertionError("source is not an internal-only mechanism artifact")

    maxima = {
        "warm_cost": 0.0,
        "proposal_cost": 0.0,
        "selected_cost": 0.0,
        "warm_floor_violation": 0.0,
        "selected_sequence": 0.0,
        "rng_state": 0.0,
        "running_state": 0.0,
    }
    proposal_selected = 0
    rows = []
    device = torch.device(args.device)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if not 0 <= args.seed_index < arrays["warm_gaussian_weighted_sequence"].shape[0]:
            raise ValueError("seed-index is out of range")
        indices = stable_indices(len(arrays["source_path"]), args.contexts)
        for index in indices:
            source_path = Path(str(arrays["source_path"][index]))
            with np.load(source_path, allow_pickle=False) as source:
                params = TorchMPPIParams(**json.loads(str(source["mppi_params_json"])))
                backend = TorchDynamicBicycleRolloutBackend(
                    TorchDBMParams(**json.loads(str(source["dbm_params_json"])))
                )
                backend.set_initial_lateral_velocity(
                    float(np.asarray(source["initial_state_six"], np.float32)[4])
                )
                controller = TorchMPPIController(
                    backend,
                    params,
                    TorchMPPICostWeights(**json.loads(str(source["cost_weights_json"]))),
                    device=device,
                )
                running = controller.get_init_state()
                running.mean_knots.copy_(
                    torch.from_numpy(np.asarray(source["sampling_mean_knots"], np.float32)).to(device)
                )
                running_before = running.mean_knots.detach().clone()
                rng_before = controller._generator.get_state().detach().clone()
                actor_knots = torch.from_numpy(
                    np.asarray(arrays["actor_center"][index], np.float32)
                ).to(device)
                actor_sequence = controller._interpolate_knots(actor_knots)
                warm_sequence = np.asarray(
                    arrays["warm_gaussian_weighted_sequence"][args.seed_index, index],
                    np.float32,
                )
                result = controller.hard_guard_action_sequence(
                    source["initial_state"],
                    source["current_action"],
                    source["history"],
                    source["reference"],
                    warm_sequence,
                    actor_sequence,
                )

            expected_warm = float(
                arrays["warm_gaussian_weighted_cost"][args.seed_index, index]
            )
            expected_proposal = float(arrays["direct_actor_cost"][index])
            expected_cost = min(expected_warm, expected_proposal)
            expected_index = int(expected_proposal < expected_warm)
            expected_sequence = (
                actor_sequence if expected_index else torch.from_numpy(warm_sequence).to(device)
            )
            one = {
                "warm_cost": abs(float(result["warm_cost"]) - expected_warm),
                "proposal_cost": abs(float(result["proposal_cost"]) - expected_proposal),
                "selected_cost": abs(float(result["selected_cost"]) - expected_cost),
                "warm_floor_violation": max(float(result["selected_cost"]) - expected_warm, 0.0),
                "selected_sequence": float(torch.max(torch.abs(
                    result["selected_action_sequence"] - expected_sequence
                ))),
                "rng_state": float(torch.max(torch.abs(
                    controller._generator.get_state().to(torch.int64)
                    - rng_before.to(torch.int64)
                ))),
                "running_state": float(torch.max(torch.abs(
                    running.mean_knots - running_before
                ))),
            }
            if int(result["selected_index"]) != expected_index:
                raise AssertionError("hard guard selected the wrong branch")
            proposal_selected += expected_index
            for name, value in one.items():
                maxima[name] = max(maxima[name], value)
            rows.append({
                "context_index": int(index),
                "source": str(source_path),
                "selected": result["selected_name"],
                "warm_cost": float(result["warm_cost"]),
                "proposal_cost": float(result["proposal_cost"]),
            })

    failures = {
        name: value for name, value in maxima.items()
        if value > (args.tolerance if name in ("warm_cost", "proposal_cost") else 1e-6)
    }
    validation = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS_HARD_GUARD_RUNTIME" if not failures else "FAIL_HARD_GUARD_RUNTIME",
        "partition": "consumed internal_selection mechanism diagnostic",
        "source_summary_sha256": sha256_file(summary_path),
        "source_evaluation_sha256": sha256_file(arrays_path),
        "context_count": len(rows),
        "seed_index": args.seed_index,
        "proposal_selected_count": proposal_selected,
        "proposal_selected_fraction": proposal_selected / len(rows),
        "maximum_error_or_violation": maxima,
        "cost_tolerance": args.tolerance,
        "failures": failures,
        "test_policy": "formal validation and test remain sealed",
        "caveats": [
            "This qualifies the hard-selection runtime primitive, not closed-loop behavior.",
            "Actor centers come from the separately qualified raw-input reconstruction chain.",
        ],
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
