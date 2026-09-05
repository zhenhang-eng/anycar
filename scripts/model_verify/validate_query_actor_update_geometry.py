#!/usr/bin/env python3
"""Independently recompute the Query Actor update-geometry diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from analyze_query_actor_update_geometry import analyze_seed, route  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_single_center_oac20to1 import sha256  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_actor_update_geometry_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def array_error(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
        return 0.0 if np.array_equal(left, right) else float("inf")
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        return 0.0 if np.array_equal(left, right) else float("inf")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0


def nested_error(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        common = set(left) & set(right)
        return max((nested_error(left[key], right[key]) for key in common), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max((nested_error(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, bool) or isinstance(right, bool) or isinstance(left, str) or isinstance(right, str):
        return 0.0 if left == right else float("inf")
    if left is None or right is None:
        return 0.0 if left is right else float("inf")
    return abs(float(left) - float(right))


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    source_summary_path = Path(manifest["source_summary"])
    source_validation_path = Path(manifest["source_validation"])
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "source_summary_hash": sha256(source_summary_path) == manifest["source_summary_sha256"],
        "source_validation_hash": sha256(source_validation_path) == manifest["source_validation_sha256"],
        "source_qualification": json.loads(source_validation_path.read_text())["qualification"] == config["sources"]["coarse77_qualification"],
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"]),
        "outer_unevaluated": not bool(summary["outer_fold_evaluated"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"]),
        "query_analytic_gradient_absent": not bool(summary["query_analytic_gradient_consumed"]),
    }
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    if sha256(Path(manifest["query_checkpoint"])) != manifest["query_checkpoint_sha256"]:
        raise AssertionError("Query checkpoint hash mismatch")
    source_summary = json.loads(source_summary_path.read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    array_errors, report_errors, recomputed_records, recomputed_arrays = {}, {}, [], []
    cost_keys = {"base_selection_cost", "step_selection_cost", "group_gain"}
    for position, seed in enumerate(config["population"]["seeds"]):
        key = str(seed)
        source_record = source_summary["records"]["coarse_to_fine77"][key]
        report, arrays = analyze_seed(int(seed), source_record, data, controller, config, device)
        saved_report = summary["records"][position]
        arrays_path = Path(saved_report["output_arrays"])
        if sha256(arrays_path) != saved_report["output_arrays_sha256"] or sha256(arrays_path) != manifest["seed_arrays_sha256"][key]:
            raise AssertionError(f"seed {seed} diagnostic arrays hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        errors = {name: array_error(saved[name], arrays[name]) for name in saved}
        if set(saved) != set(arrays):
            errors["array_key_set"] = float("inf")
        array_errors[key] = errors
        report_errors[key] = nested_error(
            {name: value for name, value in saved_report.items() if name not in ("output_arrays", "output_arrays_sha256")},
            report,
        )
        checks[f"seed_{seed}_gradient_action_arrays"] = all(
            error <= (1e-5 if name in cost_keys else 1e-6) for name, error in errors.items()
        )
        checks[f"seed_{seed}_report"] = report_errors[key] <= 1e-5
        checks[f"seed_{seed}_query_rollouts"] = saved_report["new_query_rollouts"] == 240
        recomputed_records.append(report)
        recomputed_arrays.append(arrays)
        print(f"validated seed={seed} max_array_error={max(errors.values()):.3g}", flush=True)
    recomputed_route = route(recomputed_records, config, recomputed_arrays)
    route_error = nested_error(summary["routing"], recomputed_route)
    checks["routing_recomputed"] = route_error <= 1e-8 and summary["decision"] == recomputed_route["decision"]
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_ACTOR_UPDATE_GEOMETRY_INDEPENDENT_PASS" if passed else "QUERY_ACTOR_UPDATE_GEOMETRY_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "array_max_abs_errors": array_errors, "report_max_abs_errors": report_errors,
        "routing_max_abs_error": route_error, "recomputed_routing": recomputed_route,
        "independently_replayed_query_candidates": 720,
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
