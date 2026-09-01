#!/usr/bin/env python3
"""Independently replay and validate batched validation-split DBM oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_pilot import replay_cost
from generate_dbm_direct_gt_validation import sha256_file


DEFAULT_RESULTS = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2"
)
DEFAULT_PARENT = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=DEFAULT_RESULTS)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument(
        "--expected-split", choices=("train", "validation"), default=None,
        help="Optionally require one non-test split; defaults to the result metadata.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    if summary["split"] not in ("train", "validation"):
        raise AssertionError("oracle result must be a non-test split")
    if args.expected_split is not None and summary["split"] != args.expected_split:
        raise AssertionError(
            f"oracle split {summary['split']} != expected {args.expected_split}"
        )
    if "test" not in summary["test_policy"]:
        raise AssertionError("missing sealed-test policy")
    device = torch.device(args.device)
    maximum_source_error = 0.0
    maximum_knot_interpolation_error = 0.0
    maximum_knot_cost_error = 0.0
    maximum_action_cost_error = 0.0
    maximum_parent_regression = 0.0
    minimum_parameterization_gap = float("inf")
    checked = 0
    for row in summary["rows"]:
        result_path = args.results / row["episode"] / row["snapshot"]
        parent_path = args.parent / row["episode"] / row["snapshot"]
        with np.load(result_path, allow_pickle=False) as result, np.load(
            row["source"], allow_pickle=False
        ) as source:
            if str(result["source_sha256"]) != sha256_file(Path(row["source"])):
                raise AssertionError(f"source hash mismatch: {result_path}")
            params = TorchMPPIParams(**json.loads(str(source["mppi_params_json"])))
            weights = TorchMPPICostWeights(**json.loads(str(source["cost_weights_json"])))
            backend = TorchDynamicBicycleRolloutBackend(
                TorchDBMParams(**json.loads(str(source["dbm_params_json"])))
            )
            backend.set_initial_lateral_velocity(float(source["initial_lateral_velocity"]))
            controller = TorchMPPIController(backend, params, weights, device)
            knots = torch.from_numpy(result["optimized_knots"]).to(device)
            stored_knot_actions = torch.from_numpy(result["knot_actions"]).to(device)
            actions = torch.from_numpy(result["optimized_actions"]).to(device)
            if torch.any(knots < -1.000001) or torch.any(knots > 1.000001):
                raise AssertionError(f"knot bound violation: {result_path}")
            if torch.any(actions < -1.000001) or torch.any(actions > 1.000001):
                raise AssertionError(f"action bound violation: {result_path}")
            interpolated = controller._interpolate_knots(knots)
            maximum_knot_interpolation_error = max(
                maximum_knot_interpolation_error,
                float(torch.max(torch.abs(interpolated - stored_knot_actions))),
            )
            initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
            current = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
            reference = controller._prepare_reference(source["reference"])
            knot_cost, _ = replay_cost(
                controller, backend, interpolated, initial, current, reference
            )
            action_cost, _ = replay_cost(
                controller, backend, actions, initial, current, reference
            )
            stored_knot_cost = torch.from_numpy(result["knot_cost_replay"]).to(device)
            stored_action_cost = torch.from_numpy(result["action_cost_replay"]).to(device)
            maximum_knot_cost_error = max(
                maximum_knot_cost_error,
                float(torch.max(torch.abs(knot_cost - stored_knot_cost))),
            )
            maximum_action_cost_error = max(
                maximum_action_cost_error,
                float(torch.max(torch.abs(action_cost - stored_action_cost))),
            )
            gap = float(knot_cost.min() - action_cost.min())
            minimum_parameterization_gap = min(minimum_parameterization_gap, gap)
            if gap < -1e-4:
                raise AssertionError(f"J100 exceeds J16: {result_path}")
            if parent_path.is_file():
                with np.load(parent_path, allow_pickle=False) as parent:
                    parent_knot = float(np.min(parent["knot_cost_replay"]))
                    parent_action = float(np.min(parent["action_cost_replay"]))
                maximum_parent_regression = max(
                    maximum_parent_regression,
                    float(knot_cost.min()) - parent_knot,
                    float(action_cost.min()) - parent_action,
                )
            checked += 1
            if checked == 1 or checked % 50 == 0:
                print(
                    f"[{checked:04d}/{summary['snapshot_count']:04d}] "
                    f"knot_err={maximum_knot_cost_error:.3g} "
                    f"action_err={maximum_action_cost_error:.3g}",
                    flush=True,
                )
    if checked != summary["snapshot_count"]:
        raise AssertionError("validated snapshot count differs from summary")
    validation = {
        "validated_snapshots": checked,
        "maximum_source_error": maximum_source_error,
        "maximum_knot_interpolation_error": maximum_knot_interpolation_error,
        "maximum_knot_cost_error": maximum_knot_cost_error,
        "maximum_action_cost_error": maximum_action_cost_error,
        "maximum_parent_regression": maximum_parent_regression,
        "minimum_j16_minus_j100": minimum_parameterization_gap,
        "test_policy": "test split not loaded or evaluated",
    }
    (args.results / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
