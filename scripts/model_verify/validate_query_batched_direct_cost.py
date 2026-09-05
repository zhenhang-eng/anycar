#!/usr/bin/env python3
"""Validate aligned-context Query batching against stored sequential OAC costs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import pretrain_query_single_center_actor_twin_critic as pretrain
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend
from query_batched_direct_cost import batched_direct_cost
from run_query_single_center_oac20to1 import direct_cost, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "outputs/query_mppi/query_target_coverage_mixed_init_fixed_lr1e5_oac_160round_20260903_v1"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_batched_direct_cost_validation_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    summary = json.loads((source / "summary.json").read_text())
    config = summary["contract"]
    data, replay_manifest, collection_manifest = pretrain.load_data(
        {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    )
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(
        Path(replay_manifest["query_checkpoint"]), device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    records = []
    all_error = []
    for record in summary["records"]:
        with np.load(record["arrays"], allow_pickle=False) as archive:
            rows = np.asarray(archive["selection_indices"])
            actions = np.asarray(archive["selection_round_action"])
            stored = np.asarray(archive["selection_round_cost"])
        start = time.perf_counter()
        batched = np.stack([
            batched_direct_cost(
                controller, data, rows, action, weights, context_batch_size=120
            )
            for action in actions
        ])
        batch_seconds = time.perf_counter() - start
        error = np.abs(batched.astype(np.float64) - stored.astype(np.float64))
        all_error.append(error.reshape(-1))
        stored_round = int(np.argmin(stored.mean(axis=1)))
        batched_round = int(np.argmin(batched.mean(axis=1)))
        selected = int(record["selected_round"])
        records.append({
            "seed": int(record["seed"]),
            "round_count": int(len(actions)),
            "state_count_per_round": int(len(rows)),
            "stored_selected_round": stored_round,
            "batched_selected_round": batched_round,
            "manifest_selected_round": selected,
            "selected_round_unchanged": stored_round == batched_round == selected,
            "cost_max_absolute_error": float(error.max()),
            "cost_mean_absolute_error": float(error.mean()),
            "selected_mean_cost_delta": float(
                batched[selected].mean(dtype=np.float64)
                - stored[selected].mean(dtype=np.float64)
            ),
            "all_rounds_batch_seconds": batch_seconds,
        })

    first = summary["records"][0]
    with np.load(first["arrays"], allow_pickle=False) as archive:
        benchmark_rows = np.asarray(archive["selection_indices"])
        benchmark_actions = np.asarray(archive["selection_round_action"])[
            int(first["selected_round"])
        ]
    direct_cost(
        controller, data, benchmark_rows[:4], benchmark_actions[:4], weights
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    sequential = direct_cost(
        controller, data, benchmark_rows, benchmark_actions, weights
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sequential_seconds = time.perf_counter() - start
    timing = {}
    batch_size_one = None
    for batch_size in (1, 8, 32, 120):
        batched_direct_cost(
            controller,
            data,
            benchmark_rows,
            benchmark_actions,
            weights,
            context_batch_size=batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = batched_direct_cost(
            controller,
            data,
            benchmark_rows,
            benchmark_actions,
            weights,
            context_batch_size=batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        difference = np.abs(result.astype(np.float64) - sequential.astype(np.float64))
        timing[str(batch_size)] = {
            "seconds": elapsed,
            "speedup_vs_sequential": sequential_seconds / elapsed,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }
        if batch_size == 1:
            batch_size_one = result

    combined_error = np.concatenate(all_error)
    checks = {
        "legacy_single_context_path_unchanged": bool(
            np.array_equal(batch_size_one, sequential)
        ),
        "selected_rounds_unchanged": all(
            record["selected_round_unchanged"] for record in records
        ),
        "all_round_cost_mean_absolute_error_at_most_1e-5": bool(
            combined_error.mean() <= 1e-5
        ),
        "selected_mean_cost_delta_at_most_1e-5": all(
            abs(record["selected_mean_cost_delta"]) <= 1e-5 for record in records
        ),
        "batch_120_speedup_at_least_10x": timing["120"]["speedup_vs_sequential"] >= 10,
        "sealed_source": (
            not summary.get("outer_fold_evaluated", True)
            and not summary.get("formal_validation_or_test_consumed", True)
            and not summary.get("dbm_fields_or_labels_consumed")
            and not summary.get("query_analytic_gradient_consumed", True)
        ),
    }
    passed = all(checks.values())
    report = {
        "qualification": (
            "QUERY_BATCHED_DIRECT_COST_RESULT_PRESERVING_PASS"
            if passed else "QUERY_BATCHED_DIRECT_COST_RESULT_PRESERVING_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "checks": checks,
        "records": records,
        "all_round_cost_max_absolute_error": float(combined_error.max()),
        "all_round_cost_mean_absolute_error": float(combined_error.mean()),
        "benchmark": {
            "state_count": int(len(benchmark_rows)),
            "sequential_seconds": sequential_seconds,
            "context_batch_size": timing,
        },
        "compatibility_contract": {
            "legacy_forward": "one context, N candidate actions; unchanged",
            "batched_forward": "B contexts, exactly one aligned action sequence per context",
            "checkpoint_selection_frequency": "unchanged",
            "checkpoint_selection_population": "unchanged 120-row inner fold",
            "historical_bitwise_replay": "continue using the legacy single-context path",
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "query_deployment_sha256": sha256(
            REPO_ROOT / "car_foundation/car_foundation/query_deployment.py"
        ),
        "torch_mppi_sha256": sha256(
            REPO_ROOT / "car_dynamics/car_dynamics/controllers_torch/mppi.py"
        ),
        "batched_cost_helper_sha256": sha256(
            REPO_ROOT / "scripts/model_verify/query_batched_direct_cost.py"
        ),
    }
    output.mkdir(parents=True)
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": "query-batched-direct-cost-validation-v1",
        "qualification": report["qualification"],
        "created_utc": report["created_utc"],
        "validation_sha256": sha256(output / "validation.json"),
        "validator": report["validator"],
        "validator_sha256": report["validator_sha256"],
        "source": str(source),
        "source_manifest_sha256": report["source_manifest_sha256"],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
