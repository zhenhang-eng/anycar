#!/usr/bin/env python3
"""Independently replay the stationarity-aware fresh Query FD reassessment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from build_query_expected_road_fullrank_sidecar import interpolate_knots  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402


DEFAULT_ARTIFACT = REPO_ROOT / "outputs/query_mppi/query_single_center_stationarity_reassessment_20260902_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def hadamard() -> np.ndarray:
    value = np.asarray([[1.0]], np.float64)
    while len(value) < 16:
        value = np.block([[value, value], [value, -value]])
    return value.astype(np.float32).reshape(16, 8, 2)


def make_inputs(data: dict[str, np.ndarray], normalization: dict) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(normalization)
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["critic_reference"], data["critic_current"]
    )
    count = len(history)
    return (history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
            np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
            np.zeros((count, 32), np.float32))


def query_cost(controller: TorchMPPIController, data: dict[str, np.ndarray], row: int,
               knots: np.ndarray) -> np.ndarray:
    result = controller.evaluate_action_sequences(
        data["state"][row], data["current_action"][row], data["history"][row:row + 1],
        data["reference"][row], interpolate_knots(knots),
    )
    return result["cost"].detach().cpu().numpy().astype(np.float32)


def critic_gradient(payload: dict, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
                    actions: np.ndarray, sigma: np.ndarray, device: torch.device) -> np.ndarray:
    models = []
    for twin in (1, 2):
        model = ConfigurableAbsoluteActionValueCritic().to(device)
        model.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
        model.eval()
        models.append(model)
    result = []
    for start in range(0, len(rows), 64):
        local_rows = rows[start:start + 64]
        action = torch.from_numpy(actions[start:start + len(local_rows)]).to(device)
        action.requires_grad_(True)
        values = []
        for twin, model in enumerate(models, start=1):
            z = model(
                torch.from_numpy(inputs[0][local_rows]).to(device),
                torch.from_numpy(inputs[1][local_rows]).to(device),
                torch.from_numpy(inputs[2][local_rows]).to(device), action[:, None],
            )[:, 0]
            training = payload[f"critic{twin}_training"]
            values.append(z * training["target_std"] + training["target_mean"])
        gradient = torch.autograd.grad(torch.maximum(values[0], values[1]).sum(), action)[0]
        result.append((gradient * torch.from_numpy(sigma).to(device)).detach().cpu().numpy())
    return np.concatenate(result).reshape(len(rows), 16).astype(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def main() -> None:
    artifact = DEFAULT_ARTIFACT.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir, pretrain = Path(manifest["replay"]), Path(manifest["pretrain"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "source_hashes": (
            sha256(replay_dir / "replay.npz") == manifest["replay_sha256"]
            and sha256(pretrain / "manifest.json") == manifest["pretrain_manifest_sha256"]
            and sha256(pretrain / "validation.json") == manifest["pretrain_validation_sha256"]
            and sha256(pretrain / "oof_predictions.npz") == manifest["oof_predictions_sha256"]
            and sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"]
        ),
        "artifact_hashes": (
            sha256(artifact / "summary.json") == manifest["summary_sha256"]
            and sha256(artifact / "audit.npz") == manifest["audit_sha256"]
        ),
        "sealed_boundaries": (
            not manifest["formal_validation_or_test_consumed"]
            and not manifest["dbm_fields_or_labels_consumed"]
            and not manifest["query_analytic_gradient_consumed"]
            and not manifest["actor_updated"]
        ),
    }
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(artifact / "audit.npz", allow_pickle=False) as archive:
        audit = {name: np.asarray(archive[name]) for name in archive.files}
    device = torch.device("cuda")
    query_model = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection = json.loads((Path(parent["source_collection"]) / "manifest.json").read_text())
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), TorchMPPIParams(**collection["collection"]["mppi"]),
        device="cuda",
    )
    sigma = np.asarray(config["fresh_fd"]["sigma"], np.float32).reshape(1, 2)
    signed = np.stack([sign * direction for direction in hadamard() for sign in (1.0, -1.0)])
    radius, ridge = float(config["fresh_fd"]["one_sided_radius_sigma"]), float(config["fresh_fd"]["ridge"])
    raw_error = clip_error = probe_cost_error = step_cost_error = 0.0
    true_gradient = np.empty_like(audit["true_gradient_z"])
    predicted_gradient = np.empty_like(audit["predicted_gradient_z"])
    for seed_index, seed in enumerate(audit["seeds"].astype(int)):
        centers = audit["actor_knots"][seed_index]
        for row in range(600):
            expected_raw = centers[row, None] + radius * signed * sigma
            expected_probe = np.clip(expected_raw, -1.0, 1.0).astype(np.float32)
            raw_error = max(raw_error, float(np.max(np.abs(expected_raw - audit["raw_probe_knots"][seed_index, row]))))
            clip_error = max(clip_error, float(np.max(np.abs(expected_probe - audit["probe_knots"][seed_index, row]))))
            replayed = query_cost(controller, data, row, expected_probe)
            probe_cost_error = max(probe_cost_error, float(np.max(np.abs(replayed - audit["probe_cost"][seed_index, row]))))
            x = ((expected_probe - centers[row, None]) / sigma).reshape(32, 16).astype(np.float64)
            y = np.log1p(replayed.astype(np.float64)) - np.log1p(float(audit["actor_cost"][seed_index, row]))
            true_gradient[seed_index, row] = np.linalg.solve(
                x.T @ x + ridge * np.eye(16), x.T @ y
            )
            step_replay = query_cost(controller, data, row, audit["step_knots"][seed_index, row:row + 1])[0]
            step_cost_error = max(step_cost_error, abs(float(step_replay - audit["step_cost"][seed_index, row])))
        for fold in range(5):
            rows = np.flatnonzero(data["fold_id"] == fold)
            payload = torch.load(
                pretrain / "checkpoints" / f"pretrain_fold{fold}_seed{seed}.pt",
                map_location=device, weights_only=False,
            )
            inputs = make_inputs(data, payload["normalization"])
            predicted_gradient[seed_index, rows] = critic_gradient(
                payload, inputs, rows, centers[rows], sigma, device
            )
        print(f"seed={seed} independent replay complete", flush=True)
    true_error = float(np.max(np.abs(true_gradient - audit["true_gradient_z"])))
    predicted_error = float(np.max(np.abs(predicted_gradient - audit["predicted_gradient_z"])))
    true_norm = np.linalg.norm(true_gradient, axis=2)
    predicted_norm = np.linalg.norm(predicted_gradient, axis=2)
    cosines = np.stack([cosine(predicted_gradient[s], true_gradient[s]) for s in range(3)])
    ratios = predicted_norm / np.maximum(true_norm, 1e-12)
    derived_errors = {
        "true_norm": float(np.max(np.abs(true_norm - audit["true_gradient_norm"]))),
        "predicted_norm": float(np.max(np.abs(predicted_norm - audit["predicted_gradient_norm"]))),
        "cosine": float(np.max(np.abs(cosines - audit["gradient_cosine"]))),
        "norm_ratio": float(np.max(np.abs(ratios - audit["gradient_norm_ratio"]))),
    }
    gates = config["nonflat_actor_update_gates"]
    gate_results = []
    flat_mask_error = 0
    for seed_index in range(3):
        threshold = np.quantile(true_norm[seed_index], 0.25)
        flat = true_norm[seed_index] <= threshold
        flat_mask_error += int(np.sum(flat != audit["flat_q25_mask"][seed_index]))
        nonflat = ~flat
        gate_results.append({
            "cosine_median": bool(np.median(cosines[seed_index, nonflat]) >= gates["gradient_cosine_median_minimum"]),
            "cosine_p10": bool(np.quantile(cosines[seed_index, nonflat], 0.10) >= gates["gradient_cosine_p10_minimum"]),
            "norm_ratio_min": bool(np.median(ratios[seed_index, nonflat]) >= gates["gradient_norm_ratio_median_minimum"]),
            "norm_ratio_max": bool(np.median(ratios[seed_index, nonflat]) <= gates["gradient_norm_ratio_median_maximum"]),
        })
    pass_count = int(sum(all(value.values()) for value in gate_results))
    expected_decision = (
        "STATIONARITY_AWARE_GATE_PASS_ALLOW_SMALL_CONTINUOUS_OAC_PILOT"
        if pass_count >= 2 else "GLOBAL_VALUE_PASS_ALLOW_FROZEN_ACTOR_CRITIC_BURNIN_THEN_REPEAT_FD"
    )
    checks.update({
        "probe_construction_exact": raw_error == 0.0 and clip_error == 0.0,
        "all_query_costs_replay_exact": probe_cost_error == 0.0 and step_cost_error == 0.0,
        "true_gradient_reconstruction_close": true_error <= 2e-6,
        "critic_checkpoint_gradient_close": predicted_error <= 2e-6,
        "derived_arrays_close": max(derived_errors.values()) <= 2e-6 and flat_mask_error == 0,
        "routing_consistent": pass_count == summary["nonflat_seed_pass_count"] and summary["decision"] == expected_decision,
    })
    qualification = (
        "QUERY_SINGLE_CENTER_STATIONARITY_REASSESSMENT_INDEPENDENT_PASS"
        if all(checks.values()) else "QUERY_SINGLE_CENTER_STATIONARITY_REASSESSMENT_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification, "decision": expected_decision, "checks": checks,
        "errors": {"raw_probe": raw_error, "clipped_probe": clip_error,
                   "probe_query_cost": probe_cost_error, "step_query_cost": step_cost_error,
                   "true_gradient": true_error, "predicted_gradient": predicted_error,
                   **derived_errors, "flat_mask_count": flat_mask_error},
        "nonflat_seed_pass_count": pass_count, "gate_results": gate_results,
        "independently_replayed_query_rollouts": int(3 * 600 * 33),
        "actor_updated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    (artifact / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(json.dumps(validation, indent=2, default=json_default))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
