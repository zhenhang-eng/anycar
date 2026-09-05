#!/usr/bin/env python3
"""Run the frozen Actor-centered headroom audit on promoted equal-cell160 Actors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_actor_centered_headroom import dump_json, pooled_metrics, run_seed, subset  # noqa: E402
from run_query_forward_response_landscape_pilot import evaluate, sha256  # noqa: E402
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, load_inputs  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_centered_headroom_equal_cell160_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def source_contract(config: dict) -> tuple[Path, Path, dict]:
    root = Path(config["sources"]["actor_oac_root"])
    summary_path, validation_path = root / "summary.json", root / "validation.json"
    if sha256(summary_path) != config["sources"]["actor_oac_summary_sha256"]:
        raise AssertionError("source Actor summary hash mismatch")
    if sha256(validation_path) != config["sources"]["actor_oac_validation_sha256"]:
        raise AssertionError("source Actor validation hash mismatch")
    summary = json.loads(summary_path.read_text())
    if json.loads(validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("source Actor did not independently pass")
    for key, record in summary["records"].items():
        if int(record["selected_round"]) != int(config["expected_selected_round_by_seed"][key]):
            raise AssertionError(f"source selected round changed for seed {key}")
        if sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"] or sha256(Path(record["arrays"])) != record["arrays_sha256"]:
            raise AssertionError(f"source artifact hash changed for seed {key}")
    return summary_path, validation_path, summary


def build_source_adapter(
    config: dict, source_summary: dict, data: dict[str, np.ndarray], rows: np.ndarray,
    controller: TorchMPPIController, output: Path, device: torch.device,
) -> tuple[Path, list[dict]]:
    adapter_root = output / "source_execution_adapter"
    adapter_root.mkdir()
    local = subset(data, rows)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    spec = config["source_execution_adapter"]
    reports = []
    for seed_value in config["actor_seeds"]:
        seed, key = int(seed_value), str(seed_value)
        record = source_summary["records"][key]
        checkpoint_path, arrays_path = Path(record["checkpoint"]), Path(record["arrays"])
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        selected_round = int(payload["selected_round"])
        actor = actor_from_payload(payload, "selected_actor_state_dict", device)
        actor_inputs = load_inputs(data, payload["actor_normalization"])
        actor_knots = actor_predict(actor, actor_inputs, rows, device)
        replayed_cost = np.asarray([
            evaluate(controller, local, row, actor_knots[row:row + 1], weights)[0][0]
            for row in range(len(rows))
        ], np.float32)
        with np.load(arrays_path, allow_pickle=False) as archive:
            original_rows = np.asarray(archive["selection_indices"], np.int64)
            original_action = np.asarray(archive["selection_round_action"][selected_round], np.float32)
            original_cost = np.asarray(archive["selection_round_cost"][selected_round], np.float32)
        if not np.array_equal(original_rows, rows):
            raise AssertionError(f"source selection rows changed for seed {seed}")
        action_error = float(np.max(np.abs(original_action - actor_knots)))
        if action_error != float(spec["actor_action_max_abs_error"]):
            raise AssertionError(f"source Actor action mismatch for seed {seed}")
        difference = np.abs(original_cost.astype(np.float64) - replayed_cost.astype(np.float64))
        scaled = difference / np.maximum(np.abs(original_cost.astype(np.float64)), float(spec["cost_difference_scale_floor"]))
        if float(difference.max()) > float(spec["maximum_allowed_absolute_cost_difference"]) or float(scaled.max()) > float(spec["maximum_allowed_scaled_relative_cost_difference"]):
            raise AssertionError(f"source Query execution-path difference too large for seed {seed}")
        seed_dir = adapter_root / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "checkpoint.pt").symlink_to(checkpoint_path.resolve())
        action_rounds = np.zeros((selected_round + 1, *actor_knots.shape), np.float32)
        cost_rounds = np.zeros((selected_round + 1, len(rows)), np.float32)
        action_rounds[selected_round], cost_rounds[selected_round] = actor_knots, replayed_cost
        adapter_arrays = seed_dir / "oac_arrays.npz"
        np.savez_compressed(
            adapter_arrays, selection_indices=rows, selection_round_action=action_rounds,
            selection_round_cost=cost_rounds, original_selected_cost=original_cost,
            replayed_selected_cost=replayed_cost,
        )
        reports.append({
            "seed": seed, "selected_round": selected_round,
            "original_checkpoint": str(checkpoint_path.resolve()), "original_checkpoint_sha256": sha256(checkpoint_path),
            "original_arrays": str(arrays_path.resolve()), "original_arrays_sha256": sha256(arrays_path),
            "adapter_arrays": str(adapter_arrays.resolve()), "adapter_arrays_sha256": sha256(adapter_arrays),
            "actor_action_max_abs_error": action_error,
            "cost_difference": {
                "maximum": float(difference.max()), "mean": float(difference.mean()),
                "median": float(np.median(difference)), "maximum_scaled_relative": float(scaled.max()),
                "p95": float(np.quantile(difference, 0.95)), "p95_scaled_relative": float(np.quantile(scaled, 0.95)),
            },
        })
    return adapter_root, reports


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if any((config["formal_validation_or_test_consumed"], config["dbm_fields_or_labels_consumed"],
            config["analytic_dbm_or_query_gradient_consumed"], config["actor_or_critic_trained"])):
        raise AssertionError("sealed-boundary or no-training contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source_summary_path, source_validation_path, source_summary = source_contract(config)
    data, replay_manifest, collection_manifest = pretrain.load_data({"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}})
    rows = np.flatnonzero(data["fold_id"] == int(config["split_contract"]["inner_selection_fold"]))
    if len(rows) != int(config["split_contract"]["expected_state_count"]) or np.any(data["fold_id"][rows] == int(config["split_contract"]["outer_fold"])):
        raise AssertionError("inner/outer split contract failed")
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    output.mkdir(parents=True)
    adapter_root, adapter_reports = build_source_adapter(config, source_summary, data, rows, controller, output, device)
    run_config = json.loads(json.dumps(config))
    run_config["sources"]["actor_oac"] = str(adapter_root.resolve())
    records = [run_seed(int(seed), run_config, data, rows, controller, output, device) for seed in config["actor_seeds"]]
    pooled = pooled_metrics(records, config)
    final = pooled[-1]
    aggregate = float(final["aggregate_residual_reduction"])
    majority = float(final["improved_fraction"]) > 0.5
    if aggregate < 0.02:
        decision = "STOP_BROAD_QUERY_ACTOR_EXPANSION_LOW_HEADROOM"
    elif aggregate < 0.05 or not majority:
        decision = "QUERY_ACTOR_HAS_SMALL_HEADROOM_ALLOW_SINGLE_VARIABLE_AB"
    else:
        decision = "QUERY_ACTOR_HAS_MATERIAL_HEADROOM_DIAGNOSE_ABSORPTION"
    summary = {
        "qualification": "QUERY_ACTOR_CENTERED_HEADROOM_EQUAL_CELL160_AUDIT_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "source_actor_summary": {
            "qualification": source_summary["qualification"], "decision": source_summary["decision"],
            "selected_round_by_seed": {key: int(value["selected_round"]) for key, value in source_summary["records"].items()},
        },
        "source_cost_execution_adapter": adapter_reports, "records": records,
        "pooled_round_metrics": pooled, "decision": decision,
        "decision_inputs": {"final_aggregate_residual_reduction": aggregate,
                            "final_improved_fraction": float(final["improved_fraction"]), "majority_improved": majority},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
        "actor_or_critic_trained": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-actor-centered-headroom-equal-cell160-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "source_actor_summary": str(source_summary_path.resolve()), "source_actor_summary_sha256": sha256(source_summary_path),
        "source_actor_validation": str(source_validation_path.resolve()), "source_actor_validation_sha256": sha256(source_validation_path),
        "source_execution_adapter": str(adapter_root.resolve()),
        "source_execution_adapter_arrays_sha256": {f"seed_{r['seed']}": r["adapter_arrays_sha256"] for r in adapter_reports},
        "source_replay": config["sources"]["absolute_replay"], "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"], "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(output / "summary.json"),
        "result_arrays_sha256": {f"seed_{r['seed']}": r["arrays_sha256"] for r in records},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "actor_mean_cost": pooled[0]["actor_cost"]["mean"], "best_mean_cost": final["best_cost"]["mean"],
        "aggregate_residual_reduction": aggregate,
        "paired_median_relative_reduction": final["relative_reduction"]["median"],
        "improved_fraction": final["improved_fraction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
