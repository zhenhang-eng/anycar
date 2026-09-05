#!/usr/bin/env python3
"""Independently validate the paired 39/65-candidate continuous Query AC run."""

from __future__ import annotations

import argparse
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

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_candidate_bank_ab import response_bank39, response_bank65  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    load_inputs,
    sha256,
)
from run_query_target_coverage_mixed_init_oac import warm_relative_metrics  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_candidate_bank_ab_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def max_state_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if set(left) != set(right):
        raise AssertionError("state-dict keys differ")
    errors = []
    for name in left:
        a, b = left[name].detach().cpu(), right[name].detach().cpu()
        if a.shape != b.shape:
            raise AssertionError(f"state shape differs for {name}")
        errors.append(float(torch.max(torch.abs(a - b))) if torch.is_floating_point(a) else float(not torch.equal(a, b)))
    return max(errors, default=0.0)


def array_error(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise AssertionError(f"array shape differs: {left.shape} != {right.shape}")
    if left.dtype.kind in "OUSb" or right.dtype.kind in "OUSb":
        return 0.0 if np.array_equal(left, right) else 1.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def nested_error(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("nested keys differ")
        return max((nested_error(left[name], right[name]) for name in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError("nested sequence lengths differ")
        return max((nested_error(a, b) for a, b in zip(left, right)), default=0.0)
    return array_error(np.asarray(left), np.asarray(right))


def validate_adapter(summary: dict[str, Any], source_checkpoint: Path) -> dict[str, float]:
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    actor_adapter = torch.load(summary["source_actor_adapter"], map_location="cpu", weights_only=False)
    critic_adapter = torch.load(summary["source_adapter"], map_location="cpu", weights_only=False)
    if sha256(Path(summary["source_actor_adapter"])) != summary["source_actor_adapter_sha256"]:
        raise AssertionError("actor adapter hash mismatch")
    if sha256(Path(summary["source_adapter"])) != summary["source_adapter_sha256"]:
        raise AssertionError("critic adapter hash mismatch")
    return {
        "actor": max_state_error(source["selected_actor_state_dict"], actor_adapter["selected_actor_state_dict"]),
        "critic1": max_state_error(source["selected_critic1_state_dict"], critic_adapter["critic1_state_dict"]),
        "critic2": max_state_error(source["selected_critic2_state_dict"], critic_adapter["critic2_state_dict"]),
        "actor_normalization": nested_error(
            source["actor_normalization"], actor_adapter["normalization"]
        ),
        "critic_normalization": nested_error(
            source["critic_normalization"], critic_adapter["normalization"]
        ),
    }


def validate_arm(
    arm: str,
    record: dict[str, Any],
    candidate_count: int,
    config: dict,
    data: dict[str, np.ndarray],
    controller: TorchMPPIController,
    selection: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    arrays_path = Path(record["arrays"])
    checkpoint_path = Path(record["checkpoint"])
    if sha256(arrays_path) != record["arrays_sha256"]:
        raise AssertionError(f"{arm} arrays hash mismatch")
    if sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise AssertionError(f"{arm} checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    inputs = load_inputs(data, payload["actor_normalization"])
    selected_action = actor_predict(actor, inputs, selection, device)
    selected_cost = batched_direct_cost(
        controller, data, selection, selected_action,
        {name: float(value) for name, value in config["cost_weights"].items()},
    )
    errors: dict[str, float] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if arrays["actor_batch_schedule"].shape != (40, 16, 64):
            raise AssertionError("Actor batch schedule shape changed")
        if not np.array_equal(np.unique(arrays["online_round"]), np.arange(1, 41)):
            raise AssertionError("online round coverage changed")
        group_ids = np.unique(arrays["online_group"])
        if len(group_ids) != 800:
            raise AssertionError("online group count changed")
        if len(arrays["online_cost"]) != 800 * candidate_count:
            raise AssertionError("online candidate count changed")
        errors["selected_action_reload"] = array_error(
            selected_action,
            arrays["selection_round_action"][int(record["selected_round"])],
        )
        errors["selected_cost_reload"] = array_error(
            selected_cost,
            arrays["selection_round_cost"][int(record["selected_round"])],
        )
        sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
        radii = np.linspace(
            float(config["pilot"]["probe_radius_sigma_start"]),
            float(config["pilot"]["probe_radius_sigma_end"]),
            int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
        ).astype(np.float32)[:40]
        bases = basis_bank()
        weights = {name: float(value) for name, value in config["cost_weights"].items()}
        expected_roles = np.asarray(
            ["actor"] + ["probe"] * 32 + ["response"] * 6
            + (["recenter"] * 26 if candidate_count == 65 else [])
        )
        maxima = {"action": 0.0, "cost": 0.0, "raw": 0.0, "clipped": 0.0, "role": 0.0}
        for group in group_ids:
            mask = arrays["online_group"] == group
            if int(mask.sum()) != candidate_count:
                raise AssertionError("variable-width online group")
            stored_rows = arrays["online_state_index"][mask]
            if not np.all(stored_rows == stored_rows[0]):
                raise AssertionError("online group mixes states")
            stored_round = arrays["online_round"][mask]
            if not np.all(stored_round == stored_round[0]):
                raise AssertionError("online group mixes rounds")
            row = int(stored_rows[0])
            round_index = int(stored_round[0])
            center = np.asarray(arrays["online_action"][mask][0], np.float32)
            function = response_bank39 if candidate_count == 39 else response_bank65
            expected = function(
                controller, data, row, center, float(radii[round_index - 1]),
                bases[(round_index - 1) % len(bases)], sigma, weights, config,
            )
            maxima["action"] = max(maxima["action"], array_error(arrays["online_action"][mask], expected[0]))
            maxima["cost"] = max(maxima["cost"], array_error(arrays["online_cost"][mask], expected[1]))
            maxima["raw"] = max(maxima["raw"], array_error(arrays["online_raw_action"][mask], expected[2]))
            maxima["clipped"] = max(maxima["clipped"], array_error(arrays["online_clipped"][mask], expected[3]))
            maxima["role"] = max(maxima["role"], array_error(arrays["online_role"][mask], expected_roles))
            if int(group) == 0 or (int(group) + 1) % 200 == 0:
                print(f"validate {arm} group {int(group) + 1}/800", flush=True)
        stored_report = warm_relative_metrics(
            arrays["selection_round_cost"][int(record["selected_round"])],
            data["warm_cost"][selection], data["speed_kph"][selection],
            data["variant_index"][selection],
        )
        errors["selected_metric_actor_mean"] = abs(
            float(stored_report["actor_cost"]["mean"])
            - float(record["selected"]["inner"]["actor_cost"]["mean"])
        )
        errors["selected_metric_aggregate"] = abs(
            float(stored_report["aggregate_improvement"])
            - float(record["selected"]["inner"]["aggregate_improvement"])
        )
        actor_schedule = np.asarray(arrays["actor_batch_schedule"])
        visited = np.asarray(arrays["online_state_index"]).reshape(800, candidate_count)[:, 0]
        online_round = np.asarray(arrays["online_round"]).reshape(800, candidate_count)[:, 0]
    errors.update({f"online_{name}": value for name, value in maxima.items()})
    return {
        "arm": arm,
        "errors": errors,
        "actor_batch_schedule": actor_schedule,
        "visited_rows": visited,
        "online_round": online_round,
        "all_checks_pass": all(value <= 1e-6 for value in errors.values()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path, summary_path = output / "manifest.json", output / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    if sha256(config_path) != manifest["config_sha256"]:
        raise AssertionError("config hash mismatch")
    if sha256(Path(manifest["script"])) != manifest["script_sha256"]:
        raise AssertionError("runner hash mismatch")
    if sha256(summary_path) != manifest["summary_sha256"]:
        raise AssertionError("summary hash mismatch")
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"], summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed boundary violated")

    source = Path(config["sources"]["actor_oac"])
    source_validation = source / "validation.json"
    if sha256(source_validation) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation.read_text())["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source OAC qualification changed")
    seed = int(config["pilot"]["seeds"][0])
    source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
    if sha256(source_checkpoint) != manifest["source_checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    adapter_errors = validate_adapter(summary, source_checkpoint)

    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(selection), len(outer)) != (120, 120):
        raise AssertionError("split size changed")
    if set(data["episode_id"][selection]) & set(data["episode_id"][outer]):
        raise AssertionError("inner/outer episode leakage")

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    reports = {}
    for arm, candidate_count in config["pilot"]["candidate_arms"].items():
        reports[arm] = validate_arm(
            arm, summary["records"][arm], int(candidate_count), config,
            data, controller, selection, device,
        )
    paired_schedule = array_error(
        reports["response39"]["actor_batch_schedule"],
        reports["response39_recenter26"]["actor_batch_schedule"],
    )
    paired_visited = array_error(
        reports["response39"]["visited_rows"],
        reports["response39_recenter26"]["visited_rows"],
    )
    paired_round = array_error(
        reports["response39"]["online_round"],
        reports["response39_recenter26"]["online_round"],
    )
    baseline = summary["records"]["response39"]
    treatment = summary["records"]["response39_recenter26"]
    baseline_mean = float(baseline["selected"]["inner"]["actor_cost"]["mean"])
    treatment_mean = float(treatment["selected"]["inner"]["actor_cost"]["mean"])
    baseline_aggregate = float(baseline["selected"]["inner"]["aggregate_improvement"])
    treatment_aggregate = float(treatment["selected"]["inner"]["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_RECENTER65_TO_THREE_SEED_CONTINUOUS_AC"
        if treatment_mean < baseline_mean and treatment_aggregate > baseline_aggregate
        else "RETAIN_RESPONSE39_CONTINUOUS_AC"
    )
    checks = {
        "artifact_hashes": True,
        "source_and_adapters_exact": all(value == 0.0 for value in adapter_errors.values()),
        "split_and_sealed_boundary": True,
        "paired_actor_schedule": paired_schedule == 0.0,
        "paired_context_schedule": paired_visited == 0.0 and paired_round == 0.0,
        "all_online_query_candidates_replayed": all(value["all_checks_pass"] for value in reports.values()),
        "selected_actor_and_inner_cost_reloaded": all(
            value["errors"]["selected_action_reload"] == 0.0
            and value["errors"]["selected_cost_reload"] <= 1e-6
            for value in reports.values()
        ),
        "decision_recomputed": summary["decision"] == expected_decision,
    }
    validation = {
        "qualification": (
            "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_INDEPENDENT_PASS"
            if all(checks.values()) else "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "adapter_errors": adapter_errors,
        "paired_schedule_errors": {
            "actor_batch_schedule": paired_schedule,
            "visited_rows": paired_visited,
            "online_round": paired_round,
        },
        "arm_reports": {
            arm: {"errors": value["errors"], "all_checks_pass": value["all_checks_pass"]}
            for arm, value in reports.items()
        },
        "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "baseline_selected_mean": baseline_mean,
            "treatment_selected_mean": treatment_mean,
            "treatment_mean_reduction": baseline_mean - treatment_mean,
            "baseline_warm_aggregate": baseline_aggregate,
            "treatment_warm_aggregate": treatment_aggregate,
            "treatment_aggregate_delta": treatment_aggregate - baseline_aggregate,
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
