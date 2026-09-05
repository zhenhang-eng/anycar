#!/usr/bin/env python3
"""Run paired recenter65 versus sequential coarse-to-fine77 Query AC."""

from __future__ import annotations

import argparse
import copy
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
import run_query_target_coverage_mixed_init_oac as base  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    bank_report, make_actor_adapter, make_critic_adapter, response_bank65,
)
from run_query_forward_response_landscape_pilot import evaluate, fit_response, qr_16  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_coarse_to_fine77_ab_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def response_stage(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    center: np.ndarray,
    radius: float,
    basis: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, float],
    config: dict,
    center_cost: float | None = None,
    center_residual: np.ndarray | None = None,
    include_center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if center_cost is None or center_residual is None:
        evaluated_cost, evaluated_residual = evaluate(
            controller, data, row, center[None], weights
        )
        center_cost = float(evaluated_cost[0])
        center_residual = evaluated_residual[0]
    directions = basis.reshape(16, 8, 2)
    raw_probe = np.stack([
        center + sign * radius * direction * sigma
        for direction in directions for sign in (1.0, -1.0)
    ]).astype(np.float32)
    probes = np.clip(raw_probe, -1.0, 1.0).astype(np.float32)
    probe_cost, probe_residual = evaluate(controller, data, row, probes, weights)
    fitted = fit_response(
        center, center_cost, center_residual, probes, probe_cost, probe_residual, sigma,
        float(config["pilot"]["response_fit_ridge"]),
        float(config["pilot"]["response_gauss_newton_damping"]), radius,
    )
    steps = []
    for name in ("cost_direction", "trajectory_direction", "blend_direction"):
        base_step = fitted["gn_step"] if name == "trajectory_direction" else radius * fitted[name]
        for factor in config["pilot"]["response_line_factors"]:
            steps.append(float(factor) * base_step)
    steps = np.stack(steps).reshape(6, 8, 2).astype(np.float32)
    raw_proposal = center[None] + steps * sigma
    proposals = np.clip(raw_proposal, -1.0, 1.0).astype(np.float32)
    proposal_cost, proposal_residual = evaluate(controller, data, row, proposals, weights)
    actions = np.concatenate((probes, proposals)).astype(np.float32)
    costs = np.concatenate((probe_cost, proposal_cost)).astype(np.float32)
    raw = np.concatenate((raw_probe, raw_proposal)).astype(np.float32)
    residuals = np.concatenate((probe_residual, proposal_residual)).astype(np.float32)
    if include_center:
        actions = np.concatenate((center[None], actions)).astype(np.float32)
        costs = np.concatenate((np.asarray([center_cost], np.float32), costs))
        raw = np.concatenate((center[None], raw)).astype(np.float32)
        residuals = np.concatenate((center_residual[None], residuals)).astype(np.float32)
    clipped = np.any(np.abs(raw - actions) > 1e-7, axis=(1, 2))
    return actions, costs, raw, clipped, residuals


def response_bank77(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    center: np.ndarray,
    coarse_radius: float,
    basis: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, float],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse = response_stage(
        controller, data, row, center, coarse_radius, basis, sigma, weights, config
    )
    winner = int(np.argmin(coarse[1]))
    progress = (
        float(config["pilot"]["coarse_radius_start"]) - coarse_radius
    ) / (
        float(config["pilot"]["coarse_radius_start"])
        - float(config["pilot"]["coarse_radius_end"])
    )
    fine_radius = (
        float(config["pilot"]["fine_radius_start"])
        + progress * (
            float(config["pilot"]["fine_radius_end"])
            - float(config["pilot"]["fine_radius_start"])
        )
    )
    fine = response_stage(
        controller, data, row, coarse[0][winner], fine_radius, qr_16(260905), sigma,
        weights, config, float(coarse[1][winner]), coarse[4][winner], include_center=False,
    )
    return (
        np.concatenate((coarse[0], fine[0])).astype(np.float32),
        np.concatenate((coarse[1], fine[1])).astype(np.float32),
        np.concatenate((coarse[2], fine[2])).astype(np.float32),
        np.concatenate((coarse[3], fine[3])),
    )


def rewrite_roles(record: dict, arm: str, candidate_count: int) -> None:
    path = Path(record["arrays"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    groups = len(np.unique(arrays["online_group"]))
    if arm == "control_recenter65":
        local = ["actor"] + ["probe"] * 32 + ["response"] * 6 + ["recenter"] * 26
    else:
        local = (["actor"] + ["coarse_probe"] * 32 + ["coarse_response"] * 6
                 + ["fine_probe"] * 32 + ["fine_response"] * 6)
    roles = np.tile(np.asarray(local), groups)
    if len(local) != candidate_count or len(roles) != len(arrays["online_cost"]):
        raise AssertionError("role or candidate-width contract failed")
    arrays["online_role"] = roles
    temporary = path.with_suffix(".rewrite.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    record["arrays_sha256"] = base.sha256(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if (config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]
            or config["query_analytic_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_root = Path(config["sources"]["actor_oac_root"])
    source_summary_path, source_validation_path = source_root / "summary.json", source_root / "validation.json"
    if base.sha256(source_summary_path) != config["sources"]["actor_oac_summary_sha256"]:
        raise AssertionError("source summary hash mismatch")
    if base.sha256(source_validation_path) != config["sources"]["actor_oac_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("source longrun Actor did not independently pass")
    source_summary = json.loads(source_summary_path.read_text())
    seed = int(config["pilot"]["seeds"][0])
    source_record = source_summary["records"][str(seed)]
    if int(source_record["selected_round"]) != int(config["pilot"]["source_selected_round"]):
        raise AssertionError("source selected round changed")
    source_checkpoint = Path(source_record["checkpoint"])
    if base.sha256(source_checkpoint) != source_record["checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")

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

    output.mkdir(parents=True)
    critic_adapter = make_critic_adapter(source_checkpoint, output / "source_critic_adapter", seed)
    actor_adapter = make_actor_adapter(source_checkpoint, output / "source_actor_adapter", seed)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    base.direct_cost = batched_direct_cost
    bases = base.basis_bank()
    records = {}
    for arm, candidate_count in config["pilot"]["candidate_arms"].items():
        arm_config = copy.deepcopy(config)
        arm_config["pilot"]["candidates_per_visit"] = int(candidate_count)
        arm_config["sources"]["critic_pretrain"] = str(critic_adapter.parents[1])
        arm_config["sources"]["actor_oac"] = str(actor_adapter.parents[1])
        if arm == "control_recenter65":
            start, end, function = (
                float(config["pilot"]["control_radius_start"]),
                float(config["pilot"]["control_radius_end"]), response_bank65,
            )
        else:
            start, end, function = (
                float(config["pilot"]["coarse_radius_start"]),
                float(config["pilot"]["coarse_radius_end"]), response_bank77,
            )
        arm_config["pilot"]["probe_radius_sigma_start"] = start
        arm_config["pilot"]["probe_radius_sigma_end"] = end
        radii = np.linspace(start, end, int(config["pilot"]["probe_radius_sigma_anneal_rounds"])).astype(np.float32)
        base.response_bank = function
        arm_output = output / arm
        arm_output.mkdir()
        record = base.run_seed(
            seed, arm_config, data, replay_manifest, controller, fit, selection, outer,
            radii[: int(config["pilot"]["rounds"])], bases, arm_output, device,
        )
        rewrite_roles(record, arm, int(candidate_count))
        record["bank"] = bank_report(record, int(candidate_count))
        records[arm] = record

    control, treatment = records["control_recenter65"], records["coarse_to_fine77"]
    control_mean = float(control["selected"]["inner"]["actor_cost"]["mean"])
    treatment_mean = float(treatment["selected"]["inner"]["actor_cost"]["mean"])
    control_aggregate = float(control["selected"]["inner"]["aggregate_improvement"])
    treatment_aggregate = float(treatment["selected"]["inner"]["aggregate_improvement"])
    checks = {
        "treatment_selected_mean_lower": treatment_mean < control_mean,
        "treatment_warm_aggregate_higher": treatment_aggregate > control_aggregate,
    }
    decision = "PROMOTE_COARSE_TO_FINE77_TO_THREE_SEEDS" if all(checks.values()) else "RETAIN_RECENTER65"
    comparison = {
        "control_selected_mean": control_mean,
        "treatment_selected_mean": treatment_mean,
        "treatment_mean_reduction": control_mean - treatment_mean,
        "control_warm_aggregate": control_aggregate,
        "treatment_warm_aggregate": treatment_aggregate,
        "treatment_aggregate_delta": treatment_aggregate - control_aggregate,
        "control_selected_round": int(control["selected_round"]),
        "treatment_selected_round": int(treatment["selected_round"]),
        "checks": checks,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_AB_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "source_checkpoint": str(source_checkpoint.resolve()), "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "source_actor_adapter": str(actor_adapter.resolve()), "source_actor_adapter_sha256": base.sha256(actor_adapter),
        "source_adapter": str(critic_adapter.resolve()), "source_adapter_sha256": base.sha256(critic_adapter),
        "records": records, "comparison": comparison, "decision": decision,
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-coarse-to-fine77-ab-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_actor_summary_sha256": base.sha256(source_summary_path),
        "source_actor_validation_sha256": base.sha256(source_validation_path),
        "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {arm: record["arrays_sha256"] for arm, record in records.items()},
        "result_checkpoint_sha256": {arm: record["checkpoint_sha256"] for arm, record in records.items()},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
