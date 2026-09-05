#!/usr/bin/env python3
"""Independently recompute the Query Actor speed-by-road geometry audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from analyze_query_actor_stratum_geometry import compute_seed, route  # noqa: E402
from run_query_single_center_oac20to1 import sha256  # noqa: E402
from validate_query_actor_update_geometry import array_error, nested_error  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_actor_stratum_geometry_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    geometry_manifest_path = Path(manifest["geometry_manifest"])
    geometry_summary_path = Path(manifest["geometry_summary"])
    geometry_validation_path = Path(manifest["geometry_validation"])
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "geometry_manifest_hash": sha256(geometry_manifest_path) == manifest["geometry_manifest_sha256"],
        "geometry_summary_hash": sha256(geometry_summary_path) == manifest["geometry_summary_sha256"],
        "geometry_validation_hash": sha256(geometry_validation_path) == manifest["geometry_validation_sha256"],
        "geometry_qualification": json.loads(geometry_validation_path.read_text())["qualification"] == config["sources"]["geometry_qualification"],
        "zero_new_query_rollouts": summary["new_query_rollouts"] == 0,
        "inner_not_newly_evaluated": not bool(summary["inner_newly_evaluated"]),
        "outer_unevaluated": not bool(summary["outer_fold_evaluated"]),
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"]),
        "query_analytic_gradient_absent": not bool(summary["query_analytic_gradient_consumed"]),
    }
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    geometry_manifest = json.loads(geometry_manifest_path.read_text())
    geometry_summary = json.loads(geometry_summary_path.read_text())
    geometry_config = geometry_summary["contract"]
    loader = {"outputs": {"absolute_replay": geometry_config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    road_names = [value["name"] for value in collection_manifest["collection"]["base_road_variants"]]
    checks["road_variant_semantics"] = road_names == config["population"]["variant_names"]
    device = torch.device(args.device)
    array_errors, report_errors, recomputed_records = {}, {}, []
    for position, seed in enumerate(config["population"]["seeds"]):
        stored_report = summary["records"][position]
        if int(stored_report["seed"]) != int(seed):
            raise AssertionError("stored seed order changed")
        report, arrays = compute_seed(
            int(seed), geometry_summary["records"][position], geometry_summary["records"][position],
            data, geometry_config, config, device,
        )
        arrays_path = Path(stored_report["output_arrays"])
        if sha256(arrays_path) != stored_report["output_arrays_sha256"] or sha256(arrays_path) != manifest["seed_arrays_sha256"][str(seed)]:
            raise AssertionError("diagnostic arrays hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            stored = {name: np.asarray(archive[name]) for name in archive.files}
        errors = {name: array_error(stored[name], arrays[name]) for name in stored}
        if set(stored) != set(arrays):
            errors["array_key_set"] = float("inf")
        array_errors[str(seed)] = errors
        report_errors[str(seed)] = nested_error(
            {name: value for name, value in stored_report.items() if name not in ("output_arrays", "output_arrays_sha256")},
            report,
        )
        checks[f"seed_{seed}_all_parameter_directions"] = all(value <= 1e-6 for value in errors.values())
        checks[f"seed_{seed}_report"] = report_errors[str(seed)] <= 1e-8
        checks[f"seed_{seed}_zero_rollout"] = stored_report["new_query_rollouts"] == 0
        recomputed_records.append(report)
        print(f"validated seed={seed} max_array_error={max(errors.values()):.3g}", flush=True)
        del arrays
    recomputed_route = route(recomputed_records, config)
    route_error = nested_error(summary["routing"], recomputed_route)
    checks["routing_recomputed"] = route_error <= 1e-8 and summary["decision"] == recomputed_route["decision"]
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_ACTOR_STRATUM_GEOMETRY_INDEPENDENT_PASS" if passed else "QUERY_ACTOR_STRATUM_GEOMETRY_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "array_max_abs_errors": array_errors, "report_max_abs_errors": report_errors,
        "routing_max_abs_error": route_error, "recomputed_routing": recomputed_route,
        "recomputed_parameter_direction_count": int(len(config["population"]["seeds"]) * 2 * 20),
        "new_query_rollouts": 0, "inner_newly_evaluated": False, "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
