#!/usr/bin/env python3
"""Run three-seed equal-cell Actor weighting against the validated uniform-row control."""

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
import run_query_target_coverage_mixed_init_oac as base  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_coarse_to_fine77_ab import bank_report, response_bank77, rewrite_roles  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_output_step_rms, actor_predict, actor_tensor, interpolate_state,
    physical_critic_value,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_equal_cell_ab_config_20260904_v1.json"
ACTOR_ROW_WEIGHT: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def equal_cell_actor_microsteps(
    actor: torch.nn.Module,
    selected_actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    critics: list[torch.nn.Module],
    trainings: list[dict],
    actor_inputs: tuple[np.ndarray, ...],
    critic_inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    batch_schedule: np.ndarray,
    config: dict,
    sigma: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Original gamma1 microsteps with one extra inverse-fit-cell-count factor."""
    if ACTOR_ROW_WEIGHT is None:
        raise AssertionError("Actor row weights were not initialized")
    spec = config["actor_updates"]
    gamma = float(config["actor_cost_weight_gamma"])
    maximum = float(config["actor_cost_weight_maximum"])
    before_state = copy.deepcopy(actor.state_dict())
    before_output = actor_predict(actor, actor_inputs, fit, device)
    microsteps = []
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    for microstep, rows in enumerate(batch_schedule, start=1):
        actor.train()
        action = actor_tensor(actor, actor_inputs, rows, device)
        with torch.no_grad():
            selected = actor_tensor(selected_actor, actor_inputs, rows, device)
        values = [
            physical_critic_value(critic, training, critic_inputs, rows, action, device)
            for critic, training in zip(critics, trainings)
        ]
        conservative = torch.maximum(values[0], values[1])
        unnormalized = torch.exp(gamma * conservative.detach())
        cost_weight = unnormalized.clamp(max=maximum)
        cell_weight = torch.from_numpy(ACTOR_ROW_WEIGHT[rows]).to(device=device, dtype=cost_weight.dtype)
        weight = cost_weight * cell_weight
        weight = weight / weight.mean().clamp_min(1e-12)
        sigma_tensor = torch.from_numpy(sigma).to(device).reshape(1, 1, 2)
        trust = torch.mean(torch.square((action - selected) / sigma_tensor))
        loss = torch.mean(weight * conservative) + float(
            spec["selected_checkpoint_output_trust_weight"]
        ) * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(spec["gradient_clip_norm"])
        ))
        optimizer.step()
        effective_fraction = float(
            torch.square(weight.sum()).div(len(weight) * torch.square(weight).sum()).cpu()
        )
        microsteps.append({
            "microstep": microstep,
            "loss": float(loss.detach().cpu()),
            "conservative_log_cost": float(conservative.mean().detach().cpu()),
            "selected_output_trust": float(trust.detach().cpu()),
            "gradient_norm_before_clip": gradient_norm,
            "cost_weight_effective_sample_fraction": effective_fraction,
            "cost_weight_maximum": float(weight.max().detach().cpu()),
            "cost_weight_minimum": float(weight.min().detach().cpu()),
            "cost_weight_clip_fraction": float(torch.mean((unnormalized >= maximum).float()).cpu()),
            "cell_weight_mean_before_joint_normalization": float(cell_weight.mean().detach().cpu()),
            "cell_weight_minimum": float(cell_weight.min().detach().cpu()),
            "cell_weight_maximum": float(cell_weight.max().detach().cpu()),
        })
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    end_state = copy.deepcopy(actor.state_dict())
    raw_rms = actor_output_step_rms(actor, before_output, actor_inputs, fit, sigma, device)
    cap = float(spec["cumulative_per_round_output_step_cap_sigma_rms"])
    projection = 1.0
    if raw_rms > cap:
        lower, upper = 0.0, 1.0
        for _ in range(16):
            midpoint = 0.5 * (lower + upper)
            actor.load_state_dict(interpolate_state(before_state, end_state, midpoint), strict=True)
            current = actor_output_step_rms(actor, before_output, actor_inputs, fit, sigma, device)
            if current <= cap:
                lower = midpoint
            else:
                upper = midpoint
        projection = lower
        actor.load_state_dict(interpolate_state(before_state, end_state, projection), strict=True)
    final_rms = actor_output_step_rms(actor, before_output, actor_inputs, fit, sigma, device)
    return {
        "microstep_count": int(len(batch_schedule)),
        "loss_mean": float(np.mean([item["loss"] for item in microsteps])),
        "cost_weight_effective_sample_fraction_mean": float(np.mean([
            item["cost_weight_effective_sample_fraction"] for item in microsteps
        ])),
        "cost_weight_effective_sample_fraction_minimum": float(np.min([
            item["cost_weight_effective_sample_fraction"] for item in microsteps
        ])),
        "raw_cumulative_output_step_sigma_rms": raw_rms,
        "final_cumulative_output_step_sigma_rms": final_rms,
        "cumulative_trust_projection": projection,
        "cost_weight_gamma": gamma,
        "cell_weighting": "inverse_fit_cell_count_then_minibatch_normalize",
        "microsteps": microsteps,
    }


def derive_cell_contract(data: dict[str, np.ndarray], fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    cells = sorted({(int(data["speed_kph"][row]), int(data["variant_index"][row])) for row in fit})
    if len(cells) != 20:
        raise AssertionError(f"expected 20 fit cells, found {len(cells)}")
    mapping = {cell: index for index, cell in enumerate(cells)}
    cell_id = np.full(len(data["speed_kph"]), -1, np.int16)
    for row in range(len(cell_id)):
        key = (int(data["speed_kph"][row]), int(data["variant_index"][row]))
        if key in mapping:
            cell_id[row] = mapping[key]
    counts = np.bincount(cell_id[fit], minlength=len(cells))
    if sorted(counts.tolist()) != [18] * 18 + [54] * 2:
        raise AssertionError(f"fit cell counts changed: {counts.tolist()}")
    row_weight = np.zeros(len(cell_id), np.float32)
    for index, count in enumerate(counts):
        row_weight[fit[cell_id[fit] == index]] = len(fit) / (len(cells) * int(count))
    totals = np.bincount(cell_id[fit], weights=row_weight[fit], minlength=len(cells))
    if not np.allclose(totals, np.full(len(cells), len(fit) / len(cells)), atol=1e-6):
        raise AssertionError("equal-cell expected mass check failed")
    contract = {
        "formula": "fit_count / (cell_count * fit_cell_count)",
        "fit_count": int(len(fit)), "cell_count": int(len(cells)),
        "cells": [
            {"cell_id": index, "speed_kph": speed, "variant_index": variant,
             "fit_count": int(counts[index]), "row_weight": float(len(fit) / (len(cells) * counts[index])),
             "total_expected_mass": float(totals[index])}
            for index, (speed, variant) in enumerate(cells)
        ],
        "fit_row_weight_mean": float(row_weight[fit].mean()),
        "fit_row_weight_minimum": float(row_weight[fit].min()),
        "fit_row_weight_maximum": float(row_weight[fit].max()),
    }
    return row_weight, cell_id, contract


def attach_cell_arrays(record: dict, row_weight: np.ndarray, cell_id: np.ndarray) -> None:
    path = Path(record["arrays"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["actor_fit_row_cell_weight"] = row_weight[arrays["fit_indices"]]
    arrays["fit_cell_id"] = cell_id[arrays["fit_indices"]]
    arrays["selection_cell_id"] = cell_id[arrays["selection_indices"]]
    temporary = path.with_suffix(".cell.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    record["arrays_sha256"] = base.sha256(path)


def paired_errors(control: dict, treatment: dict) -> dict[str, float]:
    with np.load(control["arrays"], allow_pickle=False) as left, np.load(
        treatment["arrays"], allow_pickle=False
    ) as right:
        control_visited = left["online_state_index"].reshape(800, 77)[:, 0]
        treatment_visited = right["online_state_index"].reshape(800, 77)[:, 0]
        control_round = left["online_round"].reshape(800, 77)[:, 0]
        treatment_round = right["online_round"].reshape(800, 77)[:, 0]
        pairs = {
            "actor_batch_schedule": (left["actor_batch_schedule"], right["actor_batch_schedule"]),
            "visited_rows": (control_visited, treatment_visited),
            "online_round": (control_round, treatment_round),
            "round0_action": (left["selection_round_action"][0], right["selection_round_action"][0]),
            "round0_cost": (left["selection_round_cost"][0], right["selection_round_cost"][0]),
        }
        return {name: float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                for name, (a, b) in pairs.items()}


def main() -> None:
    global ACTOR_ROW_WEIGHT
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if any((config["formal_validation_or_test_consumed"], config["dbm_fields_or_labels_consumed"],
            config["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    control_root = Path(config["sources"]["control_root"])
    control_summary_path, control_validation_path = control_root / "summary.json", control_root / "validation.json"
    matched_root = Path(config["sources"]["matched_step_root"])
    matched_summary_path, matched_validation_path = matched_root / "summary.json", matched_root / "validation.json"
    locked = (
        (control_summary_path, "control_summary_sha256"),
        (control_validation_path, "control_validation_sha256"),
        (matched_summary_path, "matched_step_summary_sha256"),
        (matched_validation_path, "matched_step_validation_sha256"),
    )
    for path, field in locked:
        if base.sha256(path) != config["sources"][field]:
            raise AssertionError(f"locked source hash changed: {path}")
    if json.loads(control_validation_path.read_text())["qualification"] != config["sources"]["control_qualification"]:
        raise AssertionError("control source qualification changed")
    if json.loads(matched_validation_path.read_text())["qualification"] != config["sources"]["matched_step_qualification"]:
        raise AssertionError("matched-step source qualification changed")
    control_summary = json.loads(control_summary_path.read_text())

    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    replay_validation = json.loads((Path(config["sources"]["absolute_replay"]) / "validation.json").read_text())
    if replay_validation["qualification"] != "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("source Replay did not independently pass")
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")
    ACTOR_ROW_WEIGHT, cell_id, cell_contract = derive_cell_contract(data, fit)

    output.mkdir(parents=True)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    base.direct_cost = batched_direct_cost
    base.response_bank = response_bank77
    base.mixed_actor_microsteps = equal_cell_actor_microsteps
    radii = np.linspace(
        float(config["pilot"]["coarse_radius_start"]), float(config["pilot"]["coarse_radius_end"]),
        int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
    ).astype(np.float32)[: int(config["pilot"]["rounds"])]
    bases = base.basis_bank()
    controls = copy.deepcopy(control_summary["records"]["coarse_to_fine77"])
    treatments, pairing = {}, {}
    for seed_value in config["pilot"]["seeds"]:
        seed, key = int(seed_value), str(seed_value)
        control = controls[key]
        for field in ("arrays", "checkpoint"):
            if base.sha256(Path(control[field])) != control[f"{field}_sha256"]:
                raise AssertionError(f"control seed {seed} {field} hash changed")
        adapter = control_summary["adapters"][key]
        if base.sha256(Path(adapter["actor"])) != adapter["actor_sha256"] or base.sha256(Path(adapter["critic"])) != adapter["critic_sha256"]:
            raise AssertionError(f"adapter hash changed for seed {seed}")
        arm_config = copy.deepcopy(config)
        arm_config["sources"]["actor_oac"] = str(Path(adapter["actor"]).parents[1])
        arm_config["sources"]["critic_pretrain"] = str(Path(adapter["critic"]).parents[1])
        arm_output = output / "equal_20_cell" / f"seed_{seed}_run"
        arm_output.mkdir(parents=True)
        record = base.run_seed(
            seed, arm_config, data, replay_manifest, controller, fit, selection, outer,
            radii, bases, arm_output, device,
        )
        rewrite_roles(record, "coarse_to_fine77", 77)
        attach_cell_arrays(record, ACTOR_ROW_WEIGHT, cell_id)
        record["bank"] = bank_report(record, 77)
        record["actor_cell_weighting"] = cell_contract
        treatments[key] = record
        pairing[key] = paired_errors(control, record)
        if any(value != 0.0 for value in pairing[key].values()):
            raise AssertionError(f"seed {seed} schedule/round0 pairing changed: {pairing[key]}")

    records = {"control_uniform_row": controls, "equal_20_cell": treatments}
    pooled = {
        arm: base.pooled([values[str(seed)] for seed in config["pilot"]["seeds"]], "selected", "inner", data)
        for arm, values in records.items()
    }
    per_seed, improved = {}, 0
    for seed_value in config["pilot"]["seeds"]:
        key = str(seed_value)
        control_mean = float(controls[key]["selected"]["inner"]["actor_cost"]["mean"])
        treatment_mean = float(treatments[key]["selected"]["inner"]["actor_cost"]["mean"])
        lower = treatment_mean < control_mean
        improved += int(lower)
        per_seed[key] = {
            "control_mean": control_mean, "treatment_mean": treatment_mean,
            "treatment_mean_reduction": control_mean - treatment_mean, "treatment_lower": lower,
            "control_selected_round": int(controls[key]["selected_round"]),
            "treatment_selected_round": int(treatments[key]["selected_round"]),
        }
    control_mean = float(pooled["control_uniform_row"]["actor_cost"]["mean"])
    treatment_mean = float(pooled["equal_20_cell"]["actor_cost"]["mean"])
    control_aggregate = float(pooled["control_uniform_row"]["aggregate_improvement"])
    treatment_aggregate = float(pooled["equal_20_cell"]["aggregate_improvement"])
    checks = {
        "treatment_pooled_mean_lower": treatment_mean < control_mean,
        "treatment_pooled_warm_aggregate_higher": treatment_aggregate > control_aggregate,
        "at_least_two_of_three_seed_means_lower": improved >= 2,
    }
    decision = "PROMOTE_EQUAL_20_CELL_ACTOR_OBJECTIVE" if all(checks.values()) else "RETAIN_UNIFORM_ROW_ACTOR_OBJECTIVE"
    comparison = {
        "control_pooled_mean": control_mean, "treatment_pooled_mean": treatment_mean,
        "treatment_pooled_mean_reduction": control_mean - treatment_mean,
        "control_warm_aggregate": control_aggregate, "treatment_warm_aggregate": treatment_aggregate,
        "treatment_aggregate_delta": treatment_aggregate - control_aggregate,
        "improved_seed_count": improved, "per_seed": per_seed, "checks": checks,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_EQUAL_CELL_THREE_SEED_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": records, "adapters": control_summary["adapters"], "cell_balance_contract": cell_contract,
        "paired_errors": pairing, "pooled": pooled, "comparison": comparison, "decision": decision,
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-equal-cell-three-seed-ab-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "control_summary_sha256": base.sha256(control_summary_path),
        "control_validation_sha256": base.sha256(control_validation_path),
        "matched_step_summary_sha256": base.sha256(matched_summary_path),
        "matched_step_validation_sha256": base.sha256(matched_validation_path),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "control_arrays_sha256": {seed: record["arrays_sha256"] for seed, record in controls.items()},
        "control_checkpoint_sha256": {seed: record["checkpoint_sha256"] for seed, record in controls.items()},
        "treatment_arrays_sha256": {seed: record["arrays_sha256"] for seed, record in treatments.items()},
        "treatment_checkpoint_sha256": {seed: record["checkpoint_sha256"] for seed, record in treatments.items()},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
