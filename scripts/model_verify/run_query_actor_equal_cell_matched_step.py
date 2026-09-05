#!/usr/bin/env python3
"""Compare native row-weighted and equal-cell Query Actor update directions."""

from __future__ import annotations

import argparse
import copy
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
from analyze_query_actor_update_geometry import calibrated_update  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    distribution,
    load_inputs,
    metrics,
    sha256,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_equal_cell_matched_step_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def with_variant_metrics(
    report: dict[str, Any], cost: np.ndarray, base: np.ndarray, warm: np.ndarray,
    variant: np.ndarray,
) -> dict[str, Any]:
    gain, warm_gain = base.astype(np.float64) - cost, warm.astype(np.float64) - cost
    report["by_variant_index"] = {
        str(int(value)): {
            "count": int(np.sum(variant == value)),
            "gain_vs_round0": distribution(gain[variant == value]),
            "gain_vs_warm": distribution(warm_gain[variant == value]),
        }
        for value in sorted(np.unique(variant))
    }
    return report


def compute_seed(
    seed: int, stratum_record: dict, geometry_record: dict, data: dict[str, np.ndarray],
    controller: TorchMPPIController, config: dict, device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stratum_path = Path(stratum_record["output_arrays"])
    geometry_path = Path(geometry_record["output_arrays"])
    checkpoint_path = Path(geometry_record["source_checkpoint"])
    if sha256(stratum_path) != stratum_record["output_arrays_sha256"]:
        raise AssertionError("stratum arrays hash mismatch")
    if sha256(geometry_path) != geometry_record["output_arrays_sha256"]:
        raise AssertionError("geometry arrays hash mismatch")
    if sha256(checkpoint_path) != geometry_record["source_checkpoint_sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    with np.load(stratum_path, allow_pickle=False) as archive:
        strata = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(geometry_path, allow_pickle=False) as archive:
        geometry = {name: np.asarray(archive[name]) for name in archive.files}
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fit = geometry["fit_indices"].astype(np.int64)
    selection = geometry["selection_indices"].astype(np.int64)
    outer = geometry["outer_indices_unevaluated"].astype(np.int64)
    if not np.array_equal(fit, strata["fit_indices"]):
        raise AssertionError("fit indices changed between geometry artifacts")
    actor_inputs = load_inputs(data, checkpoint["actor_normalization"])
    actor = actor_from_payload(checkpoint, "selected_actor_state_dict", device)
    base_state = copy.deepcopy(actor.state_dict())
    base_fit_action = actor_predict(actor, actor_inputs, fit, device)
    base_selection_action = actor_predict(actor, actor_inputs, selection, device)
    if np.max(np.abs(base_fit_action - geometry["base_fit_action"])) > 1e-6:
        raise AssertionError("base fit action reload mismatch")
    if np.max(np.abs(base_selection_action - geometry["base_selection_action"])) > 1e-6:
        raise AssertionError("base selection action reload mismatch")
    base_selection_cost = geometry["base_selection_cost"].astype(np.float32)
    directions = np.stack([
        geometry["critic_full_update"].astype(np.float32),
        strata["critic_stratum_update"].mean(axis=0).astype(np.float32),
    ])
    arms = list(config["population"]["arms"])
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    sigma = np.asarray(config["noise_sigma"], np.float32).reshape(1, 2)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    actions, costs, multipliers, achieved, arm_reports = [], [], [], [], {}
    for arm_index, arm in enumerate(arms):
        local_actions, local_costs, local_multiplier, local_achieved, step_reports = [], [], [], [], {}
        for step in steps:
            actor.load_state_dict(base_state, strict=True)
            multiplier, actual = calibrated_update(
                actor, base_state, base_fit_action, actor_inputs, fit, sigma,
                directions[arm_index], geometry["parameter_names"].tolist(),
                geometry["parameter_offsets"], float(step), device,
            )
            action = actor_predict(actor, actor_inputs, selection, device)
            cost = batched_direct_cost(controller, data, selection, action, weights)
            report = metrics(
                cost, base_selection_cost, data["warm_cost"][selection], data["speed_kph"][selection]
            )
            step_reports[str(float(step))] = with_variant_metrics(
                report, cost, base_selection_cost, data["warm_cost"][selection],
                data["variant_index"][selection],
            )
            local_actions.append(action)
            local_costs.append(cost)
            local_multiplier.append(multiplier)
            local_achieved.append(actual)
        arm_reports[arm] = step_reports
        actions.append(np.stack(local_actions))
        costs.append(np.stack(local_costs))
        multipliers.append(np.asarray(local_multiplier, np.float64))
        achieved.append(np.asarray(local_achieved, np.float64))
    actor.load_state_dict(base_state, strict=True)
    report = {
        "seed": seed,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": geometry_record["source_checkpoint_sha256"],
        "source_stratum_arrays": str(stratum_path),
        "source_stratum_arrays_sha256": stratum_record["output_arrays_sha256"],
        "source_geometry_arrays": str(geometry_path),
        "source_geometry_arrays_sha256": geometry_record["output_arrays_sha256"],
        "direction_cosine": float(np.dot(directions[0].astype(np.float64), directions[1].astype(np.float64)) /
                                  max(np.linalg.norm(directions[0].astype(np.float64)) * np.linalg.norm(directions[1].astype(np.float64)), 1e-30)),
        "arms": arm_reports,
        "new_query_rollouts": int(len(arms) * len(steps) * len(selection)),
        "oof_evaluated": False,
    }
    arrays = {
        "fit_indices": fit,
        "selection_indices": selection,
        "outer_indices_unevaluated": outer,
        "arm_names": np.asarray(arms),
        "matched_steps": steps,
        "parameter_names": geometry["parameter_names"],
        "parameter_offsets": geometry["parameter_offsets"],
        "parameter_update_direction": directions,
        "base_fit_action": base_fit_action,
        "base_selection_action": base_selection_action,
        "base_selection_cost": base_selection_cost,
        "step_parameter_multiplier": np.stack(multipliers),
        "step_achieved_sigma_rms": np.stack(achieved),
        "step_selection_action": np.stack(actions),
        "step_selection_cost": np.stack(costs),
    }
    return report, arrays


def route(records: list[dict[str, Any]], arrays: list[dict[str, np.ndarray]], config: dict) -> dict[str, Any]:
    arms = list(config["population"]["arms"])
    native_index, equal_index = arms.index("native_uniform_row"), arms.index("equal_20_cell")
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    step_reports, checks = {}, {}
    for step_index, step in enumerate(steps):
        native_cost = np.concatenate([value["step_selection_cost"][native_index, step_index] for value in arrays]).astype(np.float64)
        equal_cost = np.concatenate([value["step_selection_cost"][equal_index, step_index] for value in arrays]).astype(np.float64)
        base_cost = np.concatenate([value["base_selection_cost"] for value in arrays]).astype(np.float64)
        seed_wins = int(sum(
            value["step_selection_cost"][equal_index, step_index].mean()
            < value["step_selection_cost"][native_index, step_index].mean()
            for value in arrays
        ))
        key = str(float(step))
        local_checks = {
            "equal_cell_pooled_cost_lower": float(equal_cost.mean()) < float(native_cost.mean()),
            "equal_cell_at_least_two_seed_wins": seed_wins >= int(config["decision_gate"]["equal_cell_seed_win_minimum_at_each_step"]),
            "equal_cell_pooled_gain_positive": float((base_cost - equal_cost).mean()) > 0.0,
        }
        checks[key] = local_checks
        step_reports[key] = {
            "native_cost": distribution(native_cost),
            "equal_cell_cost": distribution(equal_cost),
            "equal_minus_native_cost": distribution(equal_cost - native_cost),
            "native_gain_vs_source": distribution(base_cost - native_cost),
            "equal_cell_gain_vs_source": distribution(base_cost - equal_cost),
            "equal_cell_seed_win_count": seed_wins,
            "checks": local_checks,
        }
    passed = all(value for report in checks.values() for value in report.values())
    return {
        "decision": "ADVANCE_EQUAL_CELL_TO_40ROUND_OAC_AB" if passed else "RETAIN_UNIFORM_ROW_REJECT_EQUAL_CELL",
        "all_six_gates_pass": passed,
        "by_step": step_reports,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed boundary violation")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    stratum_root = Path(config["sources"]["stratum_geometry"])
    paths = {name: stratum_root / f"{name}.json" for name in ("manifest", "summary", "validation")}
    expected = {name: config["sources"][f"stratum_{name}_sha256"] for name in paths}
    if any(sha256(path) != expected[name] for name, path in paths.items()):
        raise AssertionError("stratum source hash mismatch")
    if json.loads(paths["validation"].read_text())["qualification"] != config["sources"]["stratum_qualification"]:
        raise AssertionError("stratum source qualification changed")
    stratum_manifest = json.loads(paths["manifest"].read_text())
    stratum_summary = json.loads(paths["summary"].read_text())
    geometry_summary = json.loads(Path(stratum_manifest["geometry_summary"]).read_text())
    geometry_manifest = json.loads(Path(stratum_manifest["geometry_manifest"]).read_text())
    geometry_config = geometry_summary["contract"]
    loader = {"outputs": {"absolute_replay": geometry_config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    output.mkdir(parents=True)
    records, result_arrays, hashes = [], [], {}
    for position, seed in enumerate(config["population"]["seeds"]):
        report, arrays = compute_seed(
            int(seed), stratum_summary["records"][position], geometry_summary["records"][position],
            data, controller, config, device,
        )
        arrays_path = output / f"seed_{seed}.npz"
        np.savez_compressed(arrays_path, **arrays)
        report["output_arrays"] = str(arrays_path)
        report["output_arrays_sha256"] = sha256(arrays_path)
        hashes[str(seed)] = report["output_arrays_sha256"]
        records.append(report)
        result_arrays.append(arrays)
        print(
            f"seed={seed} cosine={report['direction_cosine']:.4f} "
            + " ".join(
                f"step={step:g} native={report['arms']['native_uniform_row'][str(float(step))]['gain_vs_round0']['mean']:.6f} "
                f"equal={report['arms']['equal_20_cell'][str(float(step))]['gain_vs_round0']['mean']:.6f}"
                for step in config["matched_output_steps_sigma_rms"]
            ), flush=True,
        )
    routing = route(records, result_arrays, config)
    summary = {
        "qualification": "QUERY_ACTOR_EQUAL_CELL_MATCHED_STEP_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": records, "routing": routing, "decision": routing["decision"],
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-actor-equal-cell-matched-step-v1", "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "stratum_manifest": str(paths["manifest"]), "stratum_manifest_sha256": sha256(paths["manifest"]),
        "stratum_summary": str(paths["summary"]), "stratum_summary_sha256": sha256(paths["summary"]),
        "stratum_validation": str(paths["validation"]), "stratum_validation_sha256": sha256(paths["validation"]),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "seed_arrays_sha256": hashes, "summary_sha256": sha256(output / "summary.json"),
        "new_query_rollouts": 1440, "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(routing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
