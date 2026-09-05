#!/usr/bin/env python3
"""Independently validate the Query Actor aggregation matched-step diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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
    load_inputs,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_actor_aggregation_microstep_20260902_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64))))


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir = Path(manifest["absolute_replay"])
    oac_dir = Path(manifest["source_oac20to1"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights_cost = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["step_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    batch_size = int(config["gradient_contract"]["batch_size"])
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(output / "summary.json") == manifest["summary_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "source_oac_manifest_hash": sha256(oac_dir / "manifest.json") == manifest["source_oac20to1_manifest_sha256"],
        "source_oac_validation_hash": sha256(oac_dir / "validation.json") == manifest["source_oac20to1_validation_sha256"],
        "query_checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "oof_not_evaluated": not bool(summary["oof_evaluated"]) and not bool(manifest["oof_evaluated"]),
        "formal_test_sealed": not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(manifest["query_analytic_gradient_consumed"]),
    }
    errors = {
        "base_fit_action": 0.0,
        "base_selection_action": 0.0,
        "base_selection_cost": 0.0,
        "conservative_log": 0.0,
        "arm_weight": 0.0,
        "aggregate_gradient": 0.0,
        "speed_gradient": 0.0,
        "step_multiplier": 0.0,
        "step_achieved": 0.0,
        "step_action": 0.0,
        "step_cost": 0.0,
        "forbidden_actor_input": 0.0,
    }
    seed_reports = []
    recomputed_pooled: dict[str, list[np.ndarray]] = {
        arm["name"]: [] for arm in config["arms"]
    }
    for record in summary["records"]:
        seed = int(record["seed"])
        arrays_path = output / f"seed_{seed}.npz"
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        checkpoint = torch.load(record["source_checkpoint"], map_location=device, weights_only=False)
        fit = saved["fit_indices"]
        selection = saved["selection_indices"]
        oof = saved["oof_indices_untouched"]
        seed_checks = {
            "arrays_hash": sha256(arrays_path) == record["output_arrays_sha256"] == manifest["seed_arrays_sha256"][str(seed)],
            "source_checkpoint_hash": sha256(Path(record["source_checkpoint"])) == record["source_checkpoint_sha256"],
            "source_arrays_hash": sha256(Path(record["source_arrays"])) == record["source_arrays_sha256"],
            "split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "split_indices_match_checkpoint": bool(
                np.array_equal(fit, checkpoint["fit_indices"])
                and np.array_equal(selection, checkpoint["selection_indices"])
                and np.array_equal(oof, checkpoint["oof_indices"])
            ),
            "split_episode_disjoint": not bool(
                set(data["episode_id"][fit].tolist()) & set(data["episode_id"][selection].tolist())
                or set(data["episode_id"][fit].tolist()) & set(data["episode_id"][oof].tolist())
                or set(data["episode_id"][selection].tolist()) & set(data["episode_id"][oof].tolist())
            ),
            "query_rollout_count": record["new_query_rollouts"] == len(selection) * len(steps) * len(config["arms"]),
            "oof_not_evaluated": not bool(record["oof_evaluated"]),
        }
        inputs = load_inputs(data, checkpoint["normalization"])
        payload = {
            "actor_training": checkpoint["actor_training"],
            "selected_actor_state_dict": checkpoint["selected_actor_state_dict"],
        }
        actor = actor_from_payload(payload, "selected_actor_state_dict", device)
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        base_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        base_fit_action = actor_predict(actor, inputs, fit, device)
        base_selection_action = actor_predict(actor, inputs, selection, device)
        base_selection_cost = direct_cost(controller, data, selection, base_selection_action, weights_cost)
        errors["base_fit_action"] = max(errors["base_fit_action"], max_error(base_fit_action, saved["base_fit_action"]))
        errors["base_selection_action"] = max(errors["base_selection_action"], max_error(base_selection_action, saved["base_selection_action"]))
        errors["base_selection_cost"] = max(errors["base_selection_cost"], max_error(base_selection_cost, saved["base_selection_cost"]))
        conservative_log = predict_conservative_log(
            actor, critics, trainings, inputs, fit, device, batch_size
        )
        errors["conservative_log"] = max(errors["conservative_log"], max_error(conservative_log, saved["fit_conservative_log_cost"]))
        names_reference = saved["parameter_names"].tolist()
        offsets_reference = saved["parameter_offsets"]
        speed_values = saved["speed_values"]
        for arm_index, arm in enumerate(config["arms"]):
            actor.load_state_dict(base_state, strict=True)
            arm_weight = objective_weights(
                conservative_log, float(arm["gamma"]), float(config["weight_maximum"])
            )
            gradient, names, offsets = actor_gradient(
                actor, critics, trainings, inputs, fit, arm_weight, device, batch_size
            )
            errors["arm_weight"] = max(errors["arm_weight"], max_error(arm_weight, saved["arm_weight"][arm_index]))
            errors["aggregate_gradient"] = max(errors["aggregate_gradient"], max_error(gradient, saved["arm_parameter_gradient"][arm_index]))
            seed_checks[f"{arm['name']}_parameter_layout"] = names == names_reference and np.array_equal(offsets, offsets_reference)
            for speed_index, speed in enumerate(speed_values):
                rows = fit[data["speed_kph"][fit] == speed]
                local_log = predict_conservative_log(
                    actor, critics, trainings, inputs, rows, device, batch_size
                )
                local_weight = objective_weights(
                    local_log, float(arm["gamma"]), float(config["weight_maximum"])
                )
                local_gradient, local_names, local_offsets = actor_gradient(
                    actor, critics, trainings, inputs, rows, local_weight, device, batch_size
                )
                errors["speed_gradient"] = max(
                    errors["speed_gradient"],
                    max_error(local_gradient, saved["arm_speed_parameter_gradient"][arm_index, speed_index]),
                )
                seed_checks[f"{arm['name']}_speed_{int(speed)}_layout"] = local_names == names_reference and np.array_equal(local_offsets, offsets_reference)
            for step_index, target in enumerate(steps):
                actor.load_state_dict(base_state, strict=True)
                multiplier, achieved, _ = calibrate_step(
                    actor, base_state, base_fit_action, inputs, fit, sigma,
                    gradient, names, offsets, float(target), device,
                )
                action = actor_predict(actor, inputs, selection, device)
                cost = direct_cost(controller, data, selection, action, weights_cost)
                errors["step_multiplier"] = max(
                    errors["step_multiplier"],
                    abs(multiplier - float(saved["matched_step_parameter_multiplier"][arm_index, step_index])),
                )
                errors["step_achieved"] = max(
                    errors["step_achieved"],
                    abs(achieved - float(saved["matched_step_achieved"][arm_index, step_index])),
                )
                errors["step_action"] = max(
                    errors["step_action"],
                    max_error(action, saved["matched_step_selection_action"][arm_index, step_index]),
                )
                errors["step_cost"] = max(
                    errors["step_cost"],
                    max_error(cost, saved["matched_step_selection_cost"][arm_index, step_index]),
                )
            primary = int(np.flatnonzero(np.isclose(steps, float(config["primary_step_sigma_rms"])))[0])
            recomputed_pooled[arm["name"]].append(
                base_selection_cost - saved["matched_step_selection_cost"][arm_index, primary]
            )
        actor.load_state_dict(base_state, strict=True)
        actor.eval()
        rows = selection[:20]
        tensors = [torch.from_numpy(value[rows]).to(device) for value in inputs]
        with torch.no_grad():
            reference = actor(*tensors)[1]
            for index in (3, 4, 5):
                changed = list(tensors)
                changed[index] = torch.randn_like(changed[index])
                errors["forbidden_actor_input"] = max(
                    errors["forbidden_actor_input"],
                    float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu()),
                )
        seed_reports.append({"seed": seed, "checks": seed_checks, "passed": bool(all(seed_checks.values()))})
    tolerances = {
        "base_fit_action": 1e-6,
        "base_selection_action": 1e-6,
        "base_selection_cost": 1e-6,
        "conservative_log": 1e-6,
        "arm_weight": 1e-6,
        "aggregate_gradient": 2e-6,
        "speed_gradient": 2e-6,
        "step_multiplier": 1e-9,
        "step_achieved": 1e-7,
        "step_action": 1e-6,
        "step_cost": 1e-6,
        "forbidden_actor_input": 1e-7,
    }
    for name, tolerance in tolerances.items():
        checks[f"{name}_within_tolerance"] = errors[name] <= tolerance
    checks["all_seed_checks"] = all(record["passed"] for record in seed_reports)
    arm0 = config["arms"][0]["name"]
    arm1 = config["arms"][1]["name"]
    pooled0 = np.concatenate(recomputed_pooled[arm0]).astype(np.float64)
    pooled1 = np.concatenate(recomputed_pooled[arm1]).astype(np.float64)
    checks["pooled_primary_reconstructed"] = all(
        abs(value - summary["pooled_primary"][arm][field]) <= 1e-6
        for arm, values in ((arm0, pooled0), (arm1, pooled1))
        for field, value in (
            ("mean", float(values.mean())),
            ("median", float(np.median(values))),
            ("p05", float(np.quantile(values, 0.05))),
        )
    )
    checks["routing_decision_reconstructed"] = summary["decision"] == "ADVANCE_GAMMA1_TO_PAIRED_CONTINUOUS_OAC20TO1" and all(summary["routing_checks"].values())
    passed = bool(all(checks.values()))
    report = {
        "qualification": "QUERY_ACTOR_AGGREGATION_MICROSTEP_INDEPENDENT_PASS" if passed else "QUERY_ACTOR_AGGREGATION_MICROSTEP_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "maximum_errors": errors,
        "tolerances": tolerances,
        "seed_reports": seed_reports,
        "decision": summary["decision"],
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
