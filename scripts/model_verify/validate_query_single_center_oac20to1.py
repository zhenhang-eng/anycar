#!/usr/bin/env python3
"""Independently validate the Query single-center continuous OAC 20:1 pilot."""

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
from run_query_forward_response_landscape_pilot import basis_bank, evaluate  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    cosine,
    direct_cost,
    distribution,
    load_inputs,
    response_bank,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_single_center_actor_visited_oac20to1_20260902_v1"


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


def reconstruct_fd(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    centers: np.ndarray,
    center_cost: np.ndarray,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    config: dict,
    weights: dict[str, float],
    sigma: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    radius = float(config["fresh_fd_diagnostic"]["one_sided_radius_sigma"])
    ridge = float(config["fresh_fd_diagnostic"]["ridge"])
    basis = basis_bank()[0].reshape(16, 8, 2)
    signed = np.stack([sign * direction for direction in basis for sign in (1.0, -1.0)])
    probes = np.empty((len(rows), 32, 8, 2), np.float32)
    costs = np.empty((len(rows), 32), np.float32)
    true_gradient = np.empty((len(rows), 16), np.float32)
    for position, row in enumerate(rows):
        local = np.clip(centers[position, None] + radius * signed * sigma, -1.0, 1.0).astype(np.float32)
        local_cost = evaluate(controller, data, int(row), local, weights)[0]
        x = ((local - centers[position, None]) / sigma).reshape(32, 16).astype(np.float64)
        y = np.log1p(local_cost.astype(np.float64)) - np.log1p(float(center_cost[position]))
        probes[position], costs[position] = local, local_cost
        true_gradient[position] = np.linalg.solve(
            x.T @ x + ridge * np.eye(16), x.T @ y
        ).astype(np.float32)
    action = torch.from_numpy(centers).to(device)
    action.requires_grad_(True)
    values = []
    for critic, training in zip(critics, trainings):
        value = critic(
            torch.from_numpy(inputs[0][rows]).to(device),
            torch.from_numpy(inputs[1][rows]).to(device),
            torch.from_numpy(inputs[2][rows]).to(device),
            action[:, None],
        )[:, 0]
        values.append(value * float(training["target_std"]) + float(training["target_mean"]))
    conservative = torch.maximum(values[0], values[1])
    predicted_gradient = (
        torch.autograd.grad(conservative.sum(), action)[0]
        * torch.from_numpy(sigma).to(device)
    ).detach().cpu().numpy().reshape(len(rows), 16).astype(np.float32)
    true_norm = np.linalg.norm(true_gradient, axis=1)
    predicted_norm = np.linalg.norm(predicted_gradient, axis=1)
    cosines = cosine(predicted_gradient, true_gradient).astype(np.float32)
    return {
        "fd_probe_knots": probes,
        "fd_probe_cost": costs,
        "fd_true_gradient_z": true_gradient,
        "fd_predicted_gradient_z": predicted_gradient,
        "fd_true_gradient_norm": true_norm.astype(np.float32),
        "fd_predicted_gradient_norm": predicted_norm.astype(np.float32),
        "fd_gradient_cosine": cosines,
        "fd_gradient_norm_ratio": (predicted_norm / np.maximum(true_norm, 1e-12)).astype(np.float32),
        "fd_flat_q25_mask": true_norm <= float(np.quantile(true_norm, 0.25)),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir = Path(manifest["absolute_replay"])
    pretrain_dir = Path(manifest["pretrain"])
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
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    rounds = int(config["pilot"]["rounds"])
    contexts = int(config["pilot"]["fit_contexts_visited_per_round"])
    candidates = int(config["pilot"]["candidates_per_visit"])
    expected_rows = rounds * contexts * candidates
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        rounds,
    ).astype(np.float32)
    bases = basis_bank()
    checks: dict[str, bool] = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(output / "summary.json") == manifest["summary_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "pretrain_manifest_hash": sha256(pretrain_dir / "manifest.json") == manifest["pretrain_manifest_sha256"],
        "stationarity_validation_hash": sha256(Path(manifest["stationarity_reassessment"]) / "validation.json") == manifest["stationarity_validation_sha256"],
        "query_checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "ratio_is_20_to_1": int(config["critic_updates"]["updates_per_actor_update_per_twin"]) == 20 and int(config["actor_updates"]["updates_per_round"]) == 1,
        "round_zero_eligible": bool(config["checkpoint_selection"]["round_zero_eligible"]),
        "fd_is_diagnostic": config["outcome_gates"]["fresh_fd_role"].startswith("diagnostic"),
        "formal_test_sealed": not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(manifest["query_analytic_gradient_consumed"]),
    }
    errors = {
        "candidate_action": 0.0,
        "candidate_raw_action": 0.0,
        "candidate_cost": 0.0,
        "selected_selection_action": 0.0,
        "selected_selection_cost": 0.0,
        "selected_oof_action": 0.0,
        "selected_oof_cost": 0.0,
        "fd_probe_action": 0.0,
        "fd_probe_cost": 0.0,
        "fd_true_gradient": 0.0,
        "fd_predicted_gradient": 0.0,
        "forbidden_actor_input": 0.0,
    }
    seed_reports = []
    for record in summary["records"]:
        seed = int(record["seed"])
        seed_dir = output / f"seed_{seed}"
        checkpoint_path = seed_dir / "checkpoint.pt"
        arrays_path = seed_dir / "pilot_arrays.npz"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        fit = saved["fit_indices"]
        selection = saved["selection_indices"]
        oof = saved["oof_indices"]
        seed_checks = {
            "checkpoint_hash": sha256(checkpoint_path) == record["checkpoint_sha256"] == manifest["seed_artifacts"][str(seed)]["checkpoint_sha256"],
            "arrays_hash": sha256(arrays_path) == record["arrays_sha256"] == manifest["seed_artifacts"][str(seed)]["arrays_sha256"],
            "seed_summary_hash": sha256(seed_dir / "summary.json") == manifest["seed_artifacts"][str(seed)]["summary_sha256"],
            "source_checkpoint_hash": sha256(Path(checkpoint["source_pretrain_checkpoint"])) == checkpoint["source_pretrain_checkpoint_sha256"],
            "nested_split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "nested_split_fold_exact": bool(
                np.all(np.isin(data["fold_id"][fit], config["split_contract"]["fit_folds"]))
                and np.all(data["fold_id"][selection] == config["split_contract"]["inner_selection_fold"])
                and np.all(data["fold_id"][oof] == config["split_contract"]["outer_fold"])
            ),
            "online_row_count": len(saved["cost"]) == expected_rows,
            "online_state_fit_only": bool(np.all(np.isin(saved["state_index"], fit))),
            "online_state_excludes_selection_oof": bool(
                not np.any(np.isin(saved["state_index"], selection))
                and not np.any(np.isin(saved["state_index"], oof))
            ),
            "candidate_finite": bool(np.all(np.isfinite(saved["action"])) and np.all(np.isfinite(saved["cost"]))),
            "candidate_bounds": bool(np.all(np.abs(saved["action"]) <= 1.0 + 1e-7)),
            "update_counts": checkpoint["actor_update_count"] == rounds and checkpoint["critic_update_count_per_twin"] == rounds * 20,
            "optimizer_serialized": all(name in checkpoint for name in (
                "actor_optimizer_state_dict", "critic1_optimizer_state_dict", "critic2_optimizer_state_dict"
            )),
            "formal_test_sealed": not bool(checkpoint["formal_validation_or_test_consumed"]),
            "dbm_fields_absent": not bool(checkpoint["dbm_fields_or_labels_consumed"]),
            "analytic_query_gradient_absent": not bool(checkpoint["query_analytic_gradient_consumed"]),
        }
        episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
        seed_checks["episode_disjoint"] = not bool(
            episode_sets[0] & episode_sets[1]
            or episode_sets[0] & episode_sets[2]
            or episode_sets[1] & episode_sets[2]
        )
        for round_index in range(1, rounds + 1):
            mask = saved["round"] == round_index
            groups = np.unique(saved["group"][mask])
            seed_checks[f"round_{round_index}_shape"] = bool(mask.sum() == contexts * candidates and len(groups) == contexts)
            rows = np.asarray([saved["state_index"][np.flatnonzero(saved["group"] == group)[0]] for group in groups])
            cells = set(zip(data["speed_index"][rows].tolist(), data["variant_index"][rows].tolist()))
            seed_checks[f"round_{round_index}_stratified"] = len(cells) == contexts
            for group in groups:
                positions = np.flatnonzero(saved["group"] == group)
                roles = saved["role"][positions]
                if not (
                    len(positions) == candidates
                    and np.sum(roles == "actor") == 1
                    and np.sum(roles == "probe") == 32
                    and np.sum(roles == "response") == 6
                ):
                    seed_checks[f"round_{round_index}_roles"] = False
                    continue
                seed_checks.setdefault(f"round_{round_index}_roles", True)
                row = int(saved["state_index"][positions[0]])
                actions, costs, raw, clipped = response_bank(
                    controller,
                    data,
                    row,
                    saved["action"][positions[0]],
                    float(radii[round_index - 1]),
                    bases[(round_index - 1) % len(bases)],
                    sigma,
                    weights,
                    config,
                )
                errors["candidate_action"] = max(errors["candidate_action"], max_error(actions, saved["action"][positions]))
                errors["candidate_raw_action"] = max(errors["candidate_raw_action"], max_error(raw, saved["raw_action"][positions]))
                errors["candidate_cost"] = max(errors["candidate_cost"], max_error(costs, saved["cost"][positions]))
                seed_checks[f"round_{round_index}_clipping"] = seed_checks.get(f"round_{round_index}_clipping", True) and bool(np.array_equal(clipped, saved["clipped"][positions]))
        source_payload = torch.load(checkpoint["source_pretrain_checkpoint"], map_location=device, weights_only=False)
        payload = dict(source_payload)
        payload["actor_training"] = checkpoint["actor_training"]
        payload["selected_actor_state_dict"] = checkpoint["selected_actor_state_dict"]
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(payload, "selected_actor_state_dict", device)
        selected_selection_action = actor_predict(actor, inputs, selection, device)
        selected_oof_action = actor_predict(actor, inputs, oof, device)
        selected_selection_cost = direct_cost(controller, data, selection, selected_selection_action, weights)
        selected_oof_cost = direct_cost(controller, data, oof, selected_oof_action, weights)
        errors["selected_selection_action"] = max(errors["selected_selection_action"], max_error(selected_selection_action, saved["selected_selection_action"]))
        errors["selected_selection_cost"] = max(errors["selected_selection_cost"], max_error(selected_selection_cost, saved["selected_selection_cost"]))
        errors["selected_oof_action"] = max(errors["selected_oof_action"], max_error(selected_oof_action, saved["selected_oof_action"]))
        errors["selected_oof_cost"] = max(errors["selected_oof_cost"], max_error(selected_oof_cost, saved["selected_oof_cost"]))
        actor.eval()
        local = selection[:20]
        tensors = [torch.from_numpy(value[local]).to(device) for value in inputs]
        with torch.no_grad():
            reference = actor(*tensors)[1]
            for index in (3, 4, 5):
                changed = list(tensors)
                changed[index] = torch.randn_like(changed[index])
                errors["forbidden_actor_input"] = max(
                    errors["forbidden_actor_input"],
                    float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu()),
                )
        selected_round = int(np.argmin(saved["selection_round_cost"].mean(axis=1)))
        seed_checks["selected_round_reconstructed"] = selected_round == checkpoint["selected_round"] == record["selected_round"]
        seed_checks["selected_selection_array_matches_round"] = max_error(
            saved["selected_selection_action"], saved["selection_round_action"][selected_round]
        ) <= 1e-6 and max_error(
            saved["selected_selection_cost"], saved["selection_round_cost"][selected_round]
        ) <= 1e-6
        critics = []
        trainings = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        recalculated_fd = reconstruct_fd(
            controller, data, oof, selected_oof_action, selected_oof_cost,
            critics, trainings, inputs, config, weights, sigma, device,
        )
        for name, error_name in (
            ("fd_probe_knots", "fd_probe_action"),
            ("fd_probe_cost", "fd_probe_cost"),
            ("fd_true_gradient_z", "fd_true_gradient"),
            ("fd_predicted_gradient_z", "fd_predicted_gradient"),
        ):
            errors[error_name] = max(errors[error_name], max_error(recalculated_fd[name], saved[name]))
        seed_checks["fd_derived_arrays"] = all(
            max_error(recalculated_fd[name], saved[name]) <= 1e-6
            for name in (
                "fd_true_gradient_norm", "fd_predicted_gradient_norm",
                "fd_gradient_cosine", "fd_gradient_norm_ratio", "fd_flat_q25_mask",
            )
        )
        seed_checks["inner_mean_gate_reconstructed"] = bool(
            saved["selected_selection_cost"].mean() <= saved["selection_round_cost"][0].mean() + 1e-6
        ) == bool(record["inner_outcome_gates"]["mean_cost_no_greater_than_round0"])
        seed_checks["inner_median_gate_reconstructed"] = bool(
            np.median(saved["selection_round_cost"][0] - saved["selected_selection_cost"]) >= -1e-6
        ) == bool(record["inner_outcome_gates"]["median_gain_vs_round0_nonnegative"])
        seed_checks["inner_p05_gate_reconstructed"] = bool(
            np.quantile(saved["selection_round_cost"][0] - saved["selected_selection_cost"], 0.05) >= -1e-6
        ) == bool(record["inner_outcome_gates"]["p05_gain_vs_round0_nonnegative"])
        seed_checks["all_round_steps_cap_only"] = all(
            float(item["actor_update"]["final_output_step_sigma_rms"])
            <= float(config["actor_updates"]["per_round_output_step_cap_sigma_rms"]) + 2e-5
            and float(item["actor_update"]["final_output_step_sigma_rms"])
            <= float(item["actor_update"]["raw_output_step_sigma_rms"]) + 1e-8
            for item in record["rounds"]
        )
        seed_reports.append({
            "seed": seed,
            "checks": seed_checks,
            "passed": bool(all(seed_checks.values())),
            "selected_round": selected_round,
            "inner_gain": distribution(saved["selection_round_cost"][0] - saved["selected_selection_cost"]),
            "oof_gain": distribution(saved["round0_oof_cost"] - saved["selected_oof_cost"]),
        })
    tolerances = {
        "candidate_action": 1e-6,
        "candidate_raw_action": 1e-6,
        "candidate_cost": 1e-6,
        "selected_selection_action": 1e-6,
        "selected_selection_cost": 1e-6,
        "selected_oof_action": 1e-6,
        "selected_oof_cost": 1e-6,
        "fd_probe_action": 1e-6,
        "fd_probe_cost": 1e-6,
        "fd_true_gradient": 1e-6,
        "fd_predicted_gradient": 1e-6,
        "forbidden_actor_input": 1e-7,
    }
    for name, tolerance in tolerances.items():
        checks[f"{name}_within_tolerance"] = errors[name] <= tolerance
    checks["all_seed_checks"] = all(value["passed"] for value in seed_reports)
    passed = bool(all(checks.values()))
    report = {
        "qualification": "QUERY_SINGLE_CENTER_OAC20TO1_INDEPENDENT_PASS" if passed else "QUERY_SINGLE_CENTER_OAC20TO1_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "maximum_errors": errors,
        "tolerances": tolerances,
        "seed_reports": seed_reports,
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
