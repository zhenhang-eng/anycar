#!/usr/bin/env python3
"""Independently replay every candidate in the high-speed strong-search oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots


DEFAULT_ROOT = Path("outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    artifact_path = Path(summary["artifact"])
    if sha256(artifact_path) != summary["artifact_sha256"]:
        raise AssertionError("artifact hash mismatch")
    for name in ("replay", "teacher"):
        if sha256(Path(summary["sources"][name])) != summary["sources"][f"{name}_sha256"]:
            raise AssertionError(f"{name} hash mismatch")
    if summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("formal validation/test unexpectedly created")
    with np.load(artifact_path, allow_pickle=False) as loaded:
        oracle = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(summary["sources"]["replay"], allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = oracle["source_indices"].astype(np.int64)
    if len(rows) != 120 or not np.all(oracle["control_step"] == 0):
        raise AssertionError("independent-state selection contract mismatch")
    if np.any(oracle["oracle_costs"] > oracle["start_costs"][:, 0] + 1e-5):
        raise AssertionError("warm floor violation")
    counts = oracle["evaluation_counts"].astype(np.int64)
    if oracle.get("path_centers", np.empty(0)).shape != (120, 5, 7, 8, 2):
        raise AssertionError("per-start six-round path is missing")
    if oracle["path_costs"].shape != (120, 5, 7):
        raise AssertionError("path cost shape mismatch")
    if np.any(np.diff(oracle["path_costs"], axis=2) > 1e-5):
        raise AssertionError("stored best-improvement path is not monotonic")
    if not np.allclose(oracle["path_centers"][:, :, 0], oracle["start_centers"]):
        raise AssertionError("path origin mismatch")
    if not np.allclose(oracle["path_centers"][:, :, -1], oracle["terminal_centers"]):
        raise AssertionError("path terminal mismatch")
    stored_min = np.asarray([
        np.min(oracle["evaluated_costs"][index, :count])
        for index, count in enumerate(counts)
    ])
    if not np.allclose(stored_min, oracle["oracle_costs"], rtol=2e-5, atol=2e-3):
        raise AssertionError("oracle is not stored-bank argmin")
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    device = torch.device(args.device)
    maximum_error = 0.0
    with torch.no_grad():
        for index, (row, count) in enumerate(zip(rows, counts)):
            centers = oracle["evaluated_centers"][index, :count]
            expected = oracle["evaluated_costs"][index, :count]
            values = []
            for start in range(0, count, args.chunk):
                knots = torch.as_tensor(centers[start : start + args.chunk][None], device=device)
                actions = interpolate_knots(knots, 50)
                values.append(batched_cost(
                    backend, weights, actions,
                    torch.as_tensor(source["state_six"][row : row + 1], device=device),
                    torch.as_tensor(source["current_action"][row : row + 1], device=device),
                    torch.as_tensor(source["reference"][row : row + 1, 1:], device=device),
                )[0].cpu().numpy())
            fresh = np.concatenate(values)
            maximum_error = max(maximum_error, float(np.max(np.abs(fresh - expected))))
            if not np.allclose(fresh, expected, rtol=2e-5, atol=2e-3):
                raise AssertionError(f"candidate replay mismatch at state {index}")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_STRONG_SEARCH_ORACLE_INDEPENDENT_REPLAY_PASS",
        "summary_sha256": sha256(summary_path), "artifact_sha256": sha256(artifact_path),
        "contexts": 120, "candidate_rollouts_replayed": int(counts.sum()),
        "maximum_dbm_cost_abs_error": maximum_error,
        "baseline_violations": 0, "formal_validation_or_test_created": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
