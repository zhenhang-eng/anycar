#!/usr/bin/env python3
"""Stationarity-aware fresh Query FD audit at all OOF Actor centers."""

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
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from build_query_expected_road_fullrank_sidecar import interpolate_knots  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_single_center_stationarity_reassessment_config_20260902_v1.json"


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


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {"count": int(len(values)), "min": float(values.min()),
            "p05": float(np.quantile(values, 0.05)), "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "p90": float(np.quantile(values, 0.90)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def hadamard_16() -> np.ndarray:
    value = np.asarray([[1.0]], np.float64)
    while len(value) < 16:
        value = np.block([[value, value], [value, -value]])
    return value.astype(np.float32)


def normalized_inputs(data: dict[str, np.ndarray], value: dict) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(value)
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["critic_reference"], data["critic_current"]
    )
    count = len(history)
    return (history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
            np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
            np.zeros((count, 32), np.float32))


def query_cost(controller: TorchMPPIController, data: dict[str, np.ndarray], row: int,
               knots: np.ndarray) -> np.ndarray:
    sequences = interpolate_knots(knots)
    result = controller.evaluate_action_sequences(
        data["state"][row], data["current_action"][row], data["history"][row:row + 1],
        data["reference"][row], sequences,
    )
    return result["cost"].detach().cpu().numpy().astype(np.float32)


def critic_gradient(payload: dict, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
                    actions: np.ndarray, sigma: np.ndarray, device: torch.device) -> np.ndarray:
    critics = []
    for twin in (1, 2):
        model = ConfigurableAbsoluteActionValueCritic().to(device)
        model.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
        model.eval()
        critics.append(model)
    output = []
    for start in range(0, len(rows), 64):
        local_rows = rows[start:start + 64]
        action = torch.from_numpy(actions[start:start + len(local_rows)]).to(device)
        action.requires_grad_(True)
        values = []
        for twin, model in enumerate(critics, start=1):
            prediction = model(
                torch.from_numpy(inputs[0][local_rows]).to(device),
                torch.from_numpy(inputs[1][local_rows]).to(device),
                torch.from_numpy(inputs[2][local_rows]).to(device), action[:, None],
            )[:, 0]
            training = payload[f"critic{twin}_training"]
            values.append(prediction * training["target_std"] + training["target_mean"])
        conservative = torch.maximum(values[0], values[1])
        gradient_abs = torch.autograd.grad(conservative.sum(), action)[0]
        output.append((gradient_abs * torch.from_numpy(sigma).to(device)).detach().cpu().numpy())
    return np.concatenate(output).reshape(len(rows), 16).astype(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    replay_dir = Path(config["sources"]["absolute_replay"])
    pretrain = Path(config["sources"]["pretrain"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    pretrain_manifest = json.loads((pretrain / "manifest.json").read_text())
    pretrain_validation = json.loads((pretrain / "validation.json").read_text())
    if pretrain_validation["qualification"] != "QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_CONFIRMED_FAIL_NO_OAC":
        raise AssertionError("unexpected pretrain qualification")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(pretrain / "oof_predictions.npz", allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]) for name in archive.files}
    seeds = saved["seeds"].astype(int)
    device = torch.device(args.device)
    query_model = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection = json.loads((Path(parent["source_collection"]) / "manifest.json").read_text())
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), TorchMPPIParams(**collection["collection"]["mppi"]),
        device=str(device),
    )
    sigma = np.asarray(config["fresh_fd"]["sigma"], np.float32).reshape(1, 2)
    radius = float(config["fresh_fd"]["one_sided_radius_sigma"])
    basis = hadamard_16().reshape(16, 8, 2)
    signed_basis = np.stack([sign * direction for direction in basis for sign in (1.0, -1.0)])
    count = len(data["row_index"])
    raw_probe = np.empty((3, count, 32, 8, 2), np.float32)
    probe = np.empty_like(raw_probe)
    probe_cost = np.empty((3, count, 32), np.float32)
    true_gradient = np.empty((3, count, 16), np.float32)
    predicted_gradient = np.empty_like(true_gradient)
    step_knots = np.empty((3, count, 8, 2), np.float32)
    step_cost = np.empty((3, count), np.float32)
    step_rms = np.empty((3, count), np.float32)
    ridge = float(config["fresh_fd"]["ridge"])
    eta = float(config["flat_contract"]["diagnostic_unscaled_step_eta"])
    cap = float(config["flat_contract"]["per_component_cap_sigma"]) * sigma
    for seed_index, seed in enumerate(seeds):
        centers = saved["actor_knots"][seed_index]
        for row in range(count):
            raw = centers[row, None] + radius * signed_basis * sigma
            local = np.clip(raw, -1.0, 1.0).astype(np.float32)
            costs = query_cost(controller, data, row, local)
            x = ((local - centers[row, None]) / sigma).reshape(32, 16).astype(np.float64)
            y = np.log1p(costs.astype(np.float64)) - np.log1p(float(saved["actor_direct_cost"][seed_index, row]))
            gradient = np.linalg.solve(x.T @ x + ridge * np.eye(16), x.T @ y)
            raw_probe[seed_index, row], probe[seed_index, row] = raw, local
            probe_cost[seed_index, row], true_gradient[seed_index, row] = costs, gradient
        for fold in range(5):
            rows = np.flatnonzero(data["fold_id"] == fold)
            payload = torch.load(
                pretrain / "checkpoints" / f"pretrain_fold{fold}_seed{seed}.pt",
                map_location=device, weights_only=False,
            )
            inputs = normalized_inputs(data, payload["normalization"])
            predicted_gradient[seed_index, rows] = critic_gradient(
                payload, inputs, rows, centers[rows], sigma, device
            )
        gradient_abs = predicted_gradient[seed_index].reshape(count, 8, 2) / sigma
        requested = -eta * gradient_abs
        bounded = np.clip(requested, -cap, cap)
        candidate = np.clip(centers + bounded, -1.0, 1.0).astype(np.float32)
        step_knots[seed_index] = candidate
        step_rms[seed_index] = np.sqrt(np.mean(np.square((candidate - centers) / sigma), axis=(1, 2)))
        for row in range(count):
            step_cost[seed_index, row] = query_cost(controller, data, row, candidate[row:row + 1])[0]
        print(f"seed={seed} fresh FD and unscaled step complete", flush=True)
    true_norm = np.linalg.norm(true_gradient, axis=2)
    predicted_norm = np.linalg.norm(predicted_gradient, axis=2)
    cosines = np.stack([cosine(predicted_gradient[s], true_gradient[s]) for s in range(3)])
    norm_ratio = predicted_norm / np.maximum(true_norm, 1e-12)
    step_gain = saved["actor_direct_cost"] - step_cost
    flat = np.zeros((3, count), bool)
    seed_reports = []
    gates = config["nonflat_actor_update_gates"]
    for seed_index, seed in enumerate(seeds):
        threshold = float(np.quantile(true_norm[seed_index], 0.25))
        flat[seed_index] = true_norm[seed_index] <= threshold
        nonflat = ~flat[seed_index]
        report = {
            "seed": int(seed), "flat_true_norm_q25_threshold": threshold,
            "all": {"true_norm": distribution(true_norm[seed_index]),
                    "predicted_norm": distribution(predicted_norm[seed_index]),
                    "cosine": distribution(cosines[seed_index]),
                    "norm_ratio": distribution(norm_ratio[seed_index]),
                    "step_gain": distribution(step_gain[seed_index]),
                    "step_rms_sigma": distribution(step_rms[seed_index])},
            "flat_q25": {"true_norm": distribution(true_norm[seed_index, flat[seed_index]]),
                         "predicted_norm": distribution(predicted_norm[seed_index, flat[seed_index]]),
                         "step_gain": distribution(step_gain[seed_index, flat[seed_index]]),
                         "step_rms_sigma": distribution(step_rms[seed_index, flat[seed_index]])},
            "nonflat_q75": {"true_norm": distribution(true_norm[seed_index, nonflat]),
                            "predicted_norm": distribution(predicted_norm[seed_index, nonflat]),
                            "cosine": distribution(cosines[seed_index, nonflat]),
                            "norm_ratio": distribution(norm_ratio[seed_index, nonflat]),
                            "step_gain": distribution(step_gain[seed_index, nonflat]),
                            "step_rms_sigma": distribution(step_rms[seed_index, nonflat])},
        }
        report["nonflat_gates"] = {
            "cosine_median": report["nonflat_q75"]["cosine"]["median"] >= gates["gradient_cosine_median_minimum"],
            "cosine_p10": report["nonflat_q75"]["cosine"]["p10"] >= gates["gradient_cosine_p10_minimum"],
            "norm_ratio_min": report["nonflat_q75"]["norm_ratio"]["median"] >= gates["gradient_norm_ratio_median_minimum"],
            "norm_ratio_max": report["nonflat_q75"]["norm_ratio"]["median"] <= gates["gradient_norm_ratio_median_maximum"],
        }
        report["nonflat_gates_pass"] = bool(all(report["nonflat_gates"].values()))
        seed_reports.append(report)
    nonflat_pass_count = int(sum(value["nonflat_gates_pass"] for value in seed_reports))
    existing_global_pass = all(
        value["metrics"]["state_pearson_log_cost"]["median"] >= 0.70
        and value["landscape_gain_recovery"] >= 0.50
        for value in pretrain_validation["pooled_oof_by_seed"]
    )
    if nonflat_pass_count >= 2:
        decision = "STATIONARITY_AWARE_GATE_PASS_ALLOW_SMALL_CONTINUOUS_OAC_PILOT"
    elif existing_global_pass:
        decision = "GLOBAL_VALUE_PASS_ALLOW_FROZEN_ACTOR_CRITIC_BURNIN_THEN_REPEAT_FD"
    else:
        decision = "GLOBAL_VALUE_FAIL_DO_NOT_START_OAC"
    output.mkdir(parents=True)
    arrays_path = output / "audit.npz"
    np.savez_compressed(
        arrays_path, seeds=seeds, row_index=data["row_index"], fold_id=data["fold_id"],
        actor_knots=saved["actor_knots"], actor_cost=saved["actor_direct_cost"],
        raw_probe_knots=raw_probe, probe_knots=probe, probe_cost=probe_cost,
        true_gradient_z=true_gradient, predicted_gradient_z=predicted_gradient,
        true_gradient_norm=true_norm, predicted_gradient_norm=predicted_norm,
        gradient_cosine=cosines, gradient_norm_ratio=norm_ratio, flat_q25_mask=flat,
        step_knots=step_knots, step_cost=step_cost, step_gain=step_gain,
        step_rms_sigma=step_rms, basis=basis, sigma=sigma.reshape(2),
    )
    summary = {
        "qualification": "QUERY_SINGLE_CENTER_STATIONARITY_REASSESSMENT_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision, "created_utc": datetime.now(timezone.utc).isoformat(),
        "existing_global_value_gate_pass": existing_global_pass,
        "nonflat_seed_pass_count": nonflat_pass_count, "seed_reports": seed_reports,
        "flat_direction_gate_applied": False,
        "fresh_query_rollouts": int(3 * count * 33),
        "actor_updated": False, "oac_actor_update_started": False,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n")
    manifest = {
        "schema_version": "query-single-center-stationarity-reassessment-v1",
        "qualification": summary["qualification"], "decision": decision,
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "replay": str(replay_dir), "replay_sha256": sha256(replay_dir / "replay.npz"),
        "pretrain": str(pretrain), "pretrain_manifest_sha256": sha256(pretrain / "manifest.json"),
        "pretrain_validation_sha256": sha256(pretrain / "validation.json"),
        "oof_predictions_sha256": sha256(pretrain / "oof_predictions.npz"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path), "audit_sha256": sha256(arrays_path),
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False, "actor_updated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "nonflat_seed_pass_count": nonflat_pass_count,
                      "seed_reports": seed_reports}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
