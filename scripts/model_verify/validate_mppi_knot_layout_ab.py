#!/usr/bin/env python3
"""Independently replay matched-search and capacity-oracle knot-layout costs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import batched_cost
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_knot_layout_search_ab import interpolation_matrix
from run_mppi_proximal_search_phase1a import (
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-dir", type=Path,
        default=Path("outputs/mppi_proposal/knot_layout_search_ab_20260825_v3"),
    )
    parser.add_argument(
        "--oracle-dir", type=Path,
        default=Path("outputs/mppi_proposal/knot_layout_capacity_oracle_20260825_v2"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/mppi_proposal/knot_layout_ab_validation_20260825_v2"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    search_summary = json.loads((args.search_dir / "summary.json").read_text())
    oracle_summary = json.loads((args.oracle_dir / "summary.json").read_text())
    search_manifest = search_summary["manifest"]
    oracle_manifest = oracle_summary["manifest"]
    if search_manifest["layouts"] != oracle_manifest["layouts"]:
        raise AssertionError("layout definitions differ")
    gt_train = Path(search_manifest["gt_train"])
    if sha256_file(gt_train / "summary.json") != search_manifest["gt_train_summary_sha256"]:
        raise AssertionError("search GT summary hash mismatch")
    if sha256_file(gt_train / "summary.json") != oracle_manifest["gt_train_summary_sha256"]:
        raise AssertionError("oracle GT summary hash mismatch")
    loader = argparse.Namespace(
        replay_labels=Path(search_manifest["replay_labels"]),
        gt_train=gt_train,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=int(search_manifest["repeat"]),
    )
    states = select_states(load_states(loader), int(search_manifest["states"]))
    search = np.load(args.search_dir / "labels.npz", allow_pickle=False)
    oracle = np.load(args.oracle_dir / "solutions.npz", allow_pickle=False)
    expected_episode = np.asarray([state["episode"] for state in states])
    expected_snapshot = np.asarray([state["snapshot"] for state in states])
    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    initial = torch.as_tensor(
        np.stack([state["initial_state_six"] for state in states]),
        dtype=torch.float32, device=device,
    )
    current = torch.as_tensor(
        np.stack([state["current_action"] for state in states]),
        dtype=torch.float32, device=device,
    )
    reference = torch.as_tensor(
        np.stack([state["reference"] for state in states]),
        dtype=torch.float32, device=device,
    )
    errors = {}
    with torch.no_grad():
        for artifact_name, loaded in (("search", search), ("oracle", oracle)):
            if not np.array_equal(loaded["episode"].astype(str), expected_episode):
                raise AssertionError(f"{artifact_name} episode order mismatch")
            if not np.array_equal(loaded["snapshot"].astype(str), expected_snapshot):
                raise AssertionError(f"{artifact_name} snapshot order mismatch")
            for name, times in search_manifest["layouts"].items():
                matrix = torch.as_tensor(
                    interpolation_matrix(np.asarray(times, np.float64)),
                    dtype=torch.float32, device=device,
                )
                knots = torch.as_tensor(
                    loaded[f"{name}_knots"], dtype=torch.float32, device=device
                )
                actions = torch.einsum("hk,bkc->bhc", matrix, knots)[:, None]
                replay = batched_cost(
                    backend, weights, actions, initial, current, reference
                )[:, 0].cpu().numpy()
                errors[f"{artifact_name}_{name}"] = float(
                    np.max(np.abs(replay - loaded[f"{name}_cost"]))
                )
    checks = {
        "layout_definitions_match": True,
        "state_order_match": True,
        "all_cost_replay_errors_le_1e_4": max(errors.values()) <= 1e-4,
        "search_train_only": search_summary["checks"]["train_only_gt"],
        "oracle_train_only": oracle_summary["checks"]["train_only_gt"],
        "formal_validation_sealed": True,
        "test_sealed": True,
    }
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "KNOT_LAYOUT_AB_VALIDATION_PASS" if all(checks.values()) else "KNOT_LAYOUT_AB_VALIDATION_FAIL",
        "manifest": {
            "search_dir": str(args.search_dir.resolve()),
            "oracle_dir": str(args.oracle_dir.resolve()),
            "search_summary_sha256": sha256_file(args.search_dir / "summary.json"),
            "oracle_summary_sha256": sha256_file(args.oracle_dir / "summary.json"),
        },
        "cost_replay_max_abs_error": errors,
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
