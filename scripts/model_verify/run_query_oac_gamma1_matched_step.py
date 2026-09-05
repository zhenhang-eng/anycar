#!/usr/bin/env python3
"""Calibrate the frozen Query OAC gamma-1 Actor direction at matched output steps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from run_query_actor_aggregation_microstep import (  # noqa: E402
    actor_gradient,
    calibrate_step,
    objective_weights,
    predict_conservative_log,
)
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    direct_cost,
    distribution,
    load_inputs,
    metrics,
    physical_critic_value,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_gamma1_matched_step_config_20260902_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def warm_relative_report(
    cost: np.ndarray,
    base_cost: np.ndarray,
    warm_cost: np.ndarray,
    speed: np.ndarray,
) -> dict[str, Any]:
    report = metrics(cost, base_cost, warm_cost, speed)
    warm_gain = warm_cost.astype(np.float64) - cost.astype(np.float64)
    report["win_or_tie_warm_fraction"] = float(np.mean(cost <= warm_cost + 1e-5))
    report["aggregate_warm_relative_improvement"] = float(
        warm_gain.sum() / max(float(warm_cost.astype(np.float64).sum()), 1e-12)
    )
    for value in sorted(np.unique(speed).tolist()):
        mask = speed == value
        report["by_speed_kph"][str(int(value))]["win_or_tie_warm_fraction"] = float(
            np.mean(cost[mask] <= warm_cost[mask] + 1e-5)
        )
        report["by_speed_kph"][str(int(value))]["aggregate_warm_relative_improvement"] = float(
            warm_gain[mask].sum() / max(float(warm_cost[mask].astype(np.float64).sum()), 1e-12)
        )
    return report


def select_source_record(summary: dict, arm: str, seed: int) -> dict:
    matches = [
        record for record in summary["records"]
        if record["arm"] == arm and int(record["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one source record for {arm} seed {seed}, got {len(matches)}")
    return matches[0]


def route(records: list[dict], steps: np.ndarray) -> tuple[str, dict[str, Any]]:
    pooled = []
    seed_positive = []
    for step_index in range(len(steps)):
        gains = np.concatenate([
            np.asarray(record["step_gain_arrays"][step_index], np.float64)
            for record in records
        ])
        pooled.append(gains)
        seed_positive.append(sum(
            record["steps"][step_index]["inner"]["gain_vs_round0"]["mean"] > 0.0
            for record in records
        ))
    pooled_means = [float(value.mean()) for value in pooled]
    strictly_increasing = all(
        pooled_means[index + 1] > pooled_means[index]
        for index in range(len(pooled_means) - 1)
    )
    best_index = int(np.argmax(pooled_means))
    any_reliable = any(value >= 2 for value in seed_positive)
    checks = {
        "pooled_gain_vs_k4_base_mean_by_step": {
            str(float(steps[index])): pooled_means[index] for index in range(len(steps))
        },
        "positive_mean_seed_count_by_step": {
            str(float(steps[index])): int(seed_positive[index]) for index in range(len(steps))
        },
        "pooled_mean_strictly_increasing": strictly_increasing,
        "best_step_sigma_rms": float(steps[best_index]),
        "best_step_positive_mean_in_at_least_2_of_3": seed_positive[best_index] >= 2,
    }
    if strictly_increasing and seed_positive[-1] >= 2:
        decision = "QUERY_GAMMA1_MEAN_RESPONSE_UNSATURATED_THROUGH_0P02"
    elif any_reliable and best_index < len(steps) - 1:
        decision = "QUERY_GAMMA1_INTERIOR_MATCHED_STEP_MEAN_OPTIMUM"
    elif any_reliable:
        decision = "QUERY_GAMMA1_MATCHED_STEP_MEAN_RESPONSE_MIXED"
    else:
        decision = "QUERY_GAMMA1_MATCHED_STEP_NO_RELIABLE_DIRECTION"
    return decision, checks


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    if config["query_analytic_gradient_consumed"]:
        raise AssertionError("Query analytic gradient is forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    source = Path(config["sources"]["gamma1_k_scan"]).resolve()
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_summary = json.loads((source / "summary.json").read_text())
    source_validation = json.loads((source / "validation.json").read_text())
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    if source_manifest["qualification"] != "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_PASS":
        raise AssertionError("source K scan manifest is not independently qualified")
    if source_validation.get("qualification") != "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_PASS":
        raise AssertionError("source K scan validation failed")
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    output.mkdir(parents=True)
    sigma = np.asarray(config["step_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    weights_cost = {name: float(value) for name, value in config["cost_weights"].items()}
    arm = str(config["population"]["arm"])
    batch_size = int(config["gradient_contract"]["batch_size"])
    gamma = float(config["actor_objective"]["gamma"])
    weight_maximum = float(config["actor_objective"]["weight_maximum"])
    records = []

    for seed_value in config["population"]["seeds"]:
        seed = int(seed_value)
        source_record = select_source_record(source_summary, arm, seed)
        checkpoint_path = Path(source_record["checkpoint"])
        arrays_path = Path(source_record["arrays"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
        fit = checkpoint["fit_indices"].astype(np.int64)
        selection = checkpoint["selection_indices"].astype(np.int64)
        oof = checkpoint["oof_indices"].astype(np.int64)
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(
            {
                "actor_training": checkpoint["actor_training"],
                "selected_actor_state_dict": checkpoint["selected_actor_state_dict"],
            },
            "selected_actor_state_dict",
            device,
        )
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        base_state = copy.deepcopy(actor.state_dict())
        base_fit_action = actor_predict(actor, inputs, fit, device)
        base_selection_action = actor_predict(actor, inputs, selection, device)
        base_selection_cost = direct_cost(controller, data, selection, base_selection_action, weights_cost)
        if np.max(np.abs(base_selection_action - source_arrays["selected_selection_action"])) > 1e-6:
            raise AssertionError("source selected Actor action mismatch")
        if np.max(np.abs(base_selection_cost - source_arrays["selected_selection_cost"])) > 1e-6:
            raise AssertionError("source selected Query cost mismatch")
        conservative_log = predict_conservative_log(
            actor, critics, trainings, inputs, fit, device, batch_size
        )
        weights = objective_weights(conservative_log, gamma, weight_maximum)
        gradient, names, offsets = actor_gradient(
            actor, critics, trainings, inputs, fit, weights, device, batch_size
        )
        step_actions, step_costs, step_multipliers, step_achieved = [], [], [], []
        step_reports, step_gain_arrays, step_warm_gain_arrays = [], [], []
        for target in steps:
            actor.load_state_dict(base_state, strict=True)
            multiplier, achieved, _ = calibrate_step(
                actor, base_state, base_fit_action, inputs, fit, sigma,
                gradient, names, offsets, float(target), device,
            )
            action = actor_predict(actor, inputs, selection, device)
            cost = direct_cost(controller, data, selection, action, weights_cost)
            report = warm_relative_report(
                cost, base_selection_cost, data["warm_cost"][selection], data["speed_kph"][selection]
            )
            step_actions.append(action)
            step_costs.append(cost)
            step_multipliers.append(multiplier)
            step_achieved.append(achieved)
            step_gain_arrays.append((base_selection_cost - cost).astype(np.float32))
            step_warm_gain_arrays.append((data["warm_cost"][selection] - cost).astype(np.float32))
            step_reports.append({
                "target_output_step_sigma_rms": float(target),
                "achieved_output_step_sigma_rms": achieved,
                "normalized_parameter_step": multiplier,
                "inner": report,
            })
        arrays_output = output / f"seed_{seed}.npz"
        np.savez_compressed(
            arrays_output,
            fit_indices=fit,
            selection_indices=selection,
            oof_indices_untouched=oof,
            parameter_names=np.asarray(names),
            parameter_offsets=offsets,
            base_fit_action=base_fit_action,
            base_selection_action=base_selection_action,
            base_selection_cost=base_selection_cost,
            fit_conservative_log_cost=conservative_log,
            fit_weight=weights,
            actor_parameter_gradient=gradient,
            matched_step_target=steps,
            matched_step_parameter_multiplier=np.asarray(step_multipliers, np.float64),
            matched_step_achieved=np.asarray(step_achieved, np.float64),
            matched_step_selection_action=np.stack(step_actions),
            matched_step_selection_cost=np.stack(step_costs),
        )
        effective_fraction = float(
            np.square(weights.astype(np.float64).sum())
            / (len(weights) * np.square(weights.astype(np.float64)).sum())
        )
        record = {
            "seed": seed,
            "source_selected_round": int(checkpoint["selected_round"]),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "source_arrays": str(arrays_path),
            "source_arrays_sha256": sha256(arrays_path),
            "output_arrays": str(arrays_output),
            "output_arrays_sha256": sha256(arrays_output),
            "weight": distribution(weights),
            "weight_effective_sample_fraction": effective_fraction,
            "weight_clip_fraction": float(np.mean(np.exp(gamma * conservative_log.astype(np.float64)) >= weight_maximum)),
            "gradient_norm": float(np.linalg.norm(gradient.astype(np.float64))),
            "base_inner": warm_relative_report(
                base_selection_cost, base_selection_cost,
                data["warm_cost"][selection], data["speed_kph"][selection],
            ),
            "steps": step_reports,
            "step_gain_arrays": [value.tolist() for value in step_gain_arrays],
            "step_warm_gain_arrays": [value.tolist() for value in step_warm_gain_arrays],
            "oof_evaluated": False,
            "new_query_rollouts": int(len(selection) * (1 + len(steps))),
        }
        records.append(record)
        print(
            f"seed={seed} selected_round={checkpoint['selected_round']} "
            + " ".join(
                f"step={steps[index]:.3f} gain={step_reports[index]['inner']['gain_vs_round0']['mean']:+.6f}"
                for index in range(len(steps))
            ),
            flush=True,
        )

    decision, routing_checks = route(records, steps)
    pooled_steps = {}
    for step_index, target in enumerate(steps):
        gain_base = np.concatenate([
            np.asarray(record["step_gain_arrays"][step_index], np.float64) for record in records
        ])
        warm_gain = np.concatenate([
            np.asarray(record["step_warm_gain_arrays"][step_index], np.float64)
            for record in records
        ])
        pooled_steps[str(float(target))] = {
            "gain_vs_k4_base": distribution(gain_base),
            "positive_mean_seed_count": int(sum(
                record["steps"][step_index]["inner"]["gain_vs_round0"]["mean"] > 0.0
                for record in records
            )),
        }
        pooled_steps[str(float(target))]["gain_vs_warm"] = distribution(warm_gain)
    # Remove raw per-state lists from the human-facing records after routing. Arrays remain in NPZ.
    for record in records:
        record.pop("step_gain_arrays")
        record.pop("step_warm_gain_arrays")
    summary = {
        "qualification": "QUERY_OAC_GAMMA1_MATCHED_STEP_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled_steps": pooled_steps,
        "routing_checks": routing_checks,
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "new_training": False,
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    helper_paths = [
        REPO_ROOT / "scripts/model_verify/run_query_actor_aggregation_microstep.py",
        REPO_ROOT / "scripts/model_verify/run_query_single_center_oac20to1.py",
    ]
    validator_path = REPO_ROOT / "scripts/model_verify/validate_query_oac_gamma1_matched_step.py"
    manifest = {
        "schema_version": "query-oac-gamma1-matched-step-v1",
        "qualification": summary["qualification"],
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "validator": str(validator_path),
        "validator_sha256": sha256(validator_path),
        "helper_sha256": {str(path): sha256(path) for path in helper_paths},
        "source_gamma1_k_scan": str(source),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_summary_sha256": sha256(source / "summary.json"),
        "source_validation_sha256": sha256(source / "validation.json"),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "seed_arrays_sha256": {str(record["seed"]): record["output_arrays_sha256"] for record in records},
        "new_training": False,
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "routing_checks": routing_checks, "pooled_steps": pooled_steps}, indent=2))


if __name__ == "__main__":
    main()
