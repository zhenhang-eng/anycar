#!/usr/bin/env python3
"""Independently reload and validate single-center Query pretraining OOF results."""

from __future__ import annotations

import argparse
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
from mppi_a2_actors import DirectNoAnchorGTXActor  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402


DEFAULT_ARTIFACT = REPO_ROOT / "outputs/query_mppi/query_single_center_actor_twin_critic_pretrain_20260902_v1"
PAIR_SAMPLES_PER_STATE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
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


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
        "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, np.float64).reshape(-1), np.asarray(right, np.float64).reshape(-1)
    if len(left) < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def inputs_from_normalizer(data: dict[str, np.ndarray], value: dict) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(value)
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["critic_reference"], data["critic_current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    )


@torch.no_grad()
def actor_predict(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
                  device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(rows), 256):
        local = rows[start:start + 256]
        tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
        output.append(model(*tensors)[1].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def critic_predict(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], data: dict[str, np.ndarray],
                   rows: np.ndarray, device: torch.device) -> np.ndarray:
    output = np.full((len(rows), data["candidate_knots"].shape[1]), np.nan, np.float32)
    model.eval()
    for count in np.unique(data["candidate_count"][rows]):
        positions = np.flatnonzero(data["candidate_count"][rows] == count)
        batch_size = 12 if int(count) == 132 else 2
        for start in range(0, len(positions), batch_size):
            local = positions[start:start + batch_size]
            global_rows = rows[local]
            output[local, :int(count)] = model(
                torch.from_numpy(inputs[0][global_rows]).to(device),
                torch.from_numpy(inputs[1][global_rows]).to(device),
                torch.from_numpy(inputs[2][global_rows]).to(device),
                torch.from_numpy(data["candidate_knots"][global_rows, :int(count)]).to(device),
            ).cpu().numpy()
    return output


def query_cost(controller: TorchMPPIController, data: dict[str, np.ndarray], rows: np.ndarray,
               knots: np.ndarray) -> np.ndarray:
    sequences = interpolate_knots(knots)
    output = np.empty(len(rows), np.float32)
    for local, row in enumerate(rows):
        result = controller.evaluate_action_sequences(
            data["state"][row], data["current_action"][row], data["history"][row:row + 1],
            data["reference"][row], sequences[local:local + 1],
        )
        output[local] = float(result["cost"][0].cpu())
    return output


def pooled_metrics(predicted: np.ndarray, data: dict[str, np.ndarray]) -> tuple[dict, float]:
    correlations, accuracies = [], []
    for row in range(600):
        valid = data["candidate_valid_mask"][row]
        prediction = predicted[row, valid].astype(np.float64)
        truth = np.log1p(data["candidate_cost"][row, valid].astype(np.float64))
        correlations.append(correlation(prediction, truth))
        rng = np.random.default_rng(1_234_000 + int(data["row_index"][row]))
        left = rng.integers(0, len(truth), PAIR_SAMPLES_PER_STATE)
        right = rng.integers(0, len(truth), PAIR_SAMPLES_PER_STATE)
        true_delta, pred_delta = truth[left] - truth[right], prediction[left] - prediction[right]
        material = np.abs(true_delta) > 1e-7
        accuracies.append(float(np.mean(np.sign(pred_delta[material]) == np.sign(true_delta[material]))))
    gains, available = [], []
    for row in np.flatnonzero(data["landscape_context_mask"]):
        eligible = data["candidate_valid_mask"][row] & data["candidate_canonical_eligible_mask"][row]
        indices = np.flatnonzero(eligible)
        selected = indices[np.argmin(predicted[row, eligible])]
        best = indices[np.argmin(data["candidate_cost"][row, eligible])]
        baseline = float(data["landscape_prior_cost"][row])
        gains.append(baseline - float(data["candidate_cost"][row, selected]))
        available.append(baseline - float(data["candidate_cost"][row, best]))
    material = np.asarray(available) > 1e-5
    recovery = float(np.sum(np.asarray(gains)[material]) / np.sum(np.asarray(available)[material]))
    return {
        "state_pearson_log_cost": distribution(np.asarray(correlations)),
        "state_pair_sign_accuracy": distribution(np.asarray(accuracies)),
    }, recovery


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir = Path(manifest["source_replay"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    replay_validation = json.loads((replay_dir / "validation.json").read_text())
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "trainer_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(artifact / "summary.json") == manifest["summary_sha256"],
        "oof_hash": sha256(artifact / "oof_predictions.npz") == manifest["oof_predictions_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["source_replay_sha256"],
        "replay_qualified": replay_validation["qualification"] == "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
        "query_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "sealed_boundaries": not manifest["formal_validation_or_test_consumed"]
        and not manifest["dbm_fields_or_labels_consumed"] and not manifest["query_analytic_gradient_consumed"],
    }
    checks["checkpoint_hashes"] = all(
        sha256(artifact / "checkpoints" / name) == digest
        for name, digest in manifest["checkpoint_sha256"].items()
    )
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    landscape = Path(replay_manifest["landscape_source"])
    with np.load(landscape / "landscape.npz", allow_pickle=False) as archive:
        land = {name: np.asarray(archive[name]) for name in archive.files}
    mapped = np.flatnonzero(data["landscape_context_mask"])
    audit = data["landscape_source_audit_row"][mapped]
    prior_branch = int(np.flatnonzero(land["branch_name"] == "prior_noanchor_canonical")[0])
    prior_cost = np.full(600, np.nan, np.float32)
    prior_cost[mapped] = land["center_cost"][audit, prior_branch, 0]
    data["landscape_prior_cost"] = prior_cost
    with np.load(artifact / "oof_predictions.npz", allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]) for name in archive.files}
    device = torch.device(args.device)
    query_model = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection = json.loads((Path(parent["source_collection"]) / "manifest.json").read_text())
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), TorchMPPIParams(**collection["collection"]["mppi"]),
        device=str(device),
    )
    actor_error, actor_cost_error, critic_error = 0.0, 0.0, 0.0
    invariance_error, split_exact = 0.0, True
    seeds = saved["seeds"].astype(int).tolist()
    for fold in range(5):
        oof = np.flatnonzero(data["fold_id"] == fold)
        expected_selection = np.flatnonzero(data["fold_id"] == ((fold + 1) % 5))
        expected_fit = np.flatnonzero(
            (data["fold_id"] != fold) & (data["fold_id"] != ((fold + 1) % 5))
        )
        for seed_index, seed in enumerate(seeds):
            path = artifact / "checkpoints" / f"pretrain_fold{fold}_seed{seed}.pt"
            payload = torch.load(path, map_location=device, weights_only=False)
            split_exact &= (
                np.array_equal(payload["oof_indices"], oof)
                and np.array_equal(payload["selection_indices"], expected_selection)
                and np.array_equal(payload["fit_indices"], expected_fit)
            )
            inputs = inputs_from_normalizer(data, payload["normalization"])
            center = torch.tensor(payload["actor_training"]["out_center"], dtype=torch.float32, device=device)
            scale = torch.tensor(payload["actor_training"]["out_scale"], dtype=torch.float32, device=device)
            actor = DirectNoAnchorGTXActor(dropout=0.0, center=center, scale=scale).to(device)
            actor.load_state_dict(payload["actor_state_dict"], strict=True)
            actor_knots = actor_predict(actor, inputs, oof, device)
            actor_error = max(actor_error, float(np.max(np.abs(actor_knots - saved["actor_knots"][seed_index, oof]))))
            replayed_cost = query_cost(controller, data, oof, actor_knots)
            actor_cost_error = max(actor_cost_error, float(np.max(np.abs(replayed_cost - saved["actor_direct_cost"][seed_index, oof]))))
            tensors = [torch.from_numpy(value[oof[:20]]).to(device) for value in inputs]
            with torch.no_grad():
                base = actor(*tensors)[1]
                for index in (3, 4, 5):
                    changed = list(tensors)
                    changed[index] = torch.randn_like(changed[index])
                    invariance_error = max(invariance_error, float(torch.max(torch.abs(actor(*changed)[1] - base)).cpu()))
            physical = []
            for twin in (1, 2):
                critic = ConfigurableAbsoluteActionValueCritic().to(device)
                critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
                prediction = critic_predict(critic, inputs, data, oof, device)
                training = payload[f"critic{twin}_training"]
                physical.append(prediction * training["target_std"] + training["target_mean"])
            conservative = np.fmax(physical[0], physical[1])
            valid = data["candidate_valid_mask"][oof]
            critic_error = max(critic_error, float(np.max(np.abs(
                conservative[valid] - saved["twin_conservative_log_cost"][seed_index, oof][valid]
            ))))
    checks.update({
        "nested_splits_exact": bool(split_exact),
        "actor_checkpoint_reload_exact": actor_error == 0.0,
        "critic_checkpoint_reload_close": critic_error <= 2e-6,
        "actor_query_replay_exact": actor_cost_error <= 1e-6,
        "strict_no_anchor_invariance": invariance_error == 0.0,
    })
    pooled = []
    thresholds = config["critic_pretrain_gates"]
    for seed_index, seed in enumerate(seeds):
        metrics, recovery = pooled_metrics(saved["twin_conservative_log_cost"][seed_index], data)
        passed = (
            metrics["state_pearson_log_cost"]["median"] >= thresholds["oof_log_cost_pearson_median_minimum"]
            and metrics["state_pair_sign_accuracy"]["median"] >= thresholds["oof_same_state_pair_sign_accuracy_median_minimum"]
            and recovery >= thresholds["oof_landscape_bank_gain_recovery_median_minimum"]
        )
        pooled.append({"seed": seed, "metrics": metrics, "landscape_gain_recovery": recovery,
                       "offline_gates_pass": bool(passed)})
    pass_count = int(sum(item["offline_gates_pass"] for item in pooled))
    checks["pooled_metrics_match_summary"] = all(
        abs(item["metrics"]["state_pearson_log_cost"]["median"] - summary["pooled_oof_by_seed"][i]["metrics"]["state_pearson_log_cost"]["median"]) <= 1e-12
        and abs(item["metrics"]["state_pair_sign_accuracy"]["median"] - summary["pooled_oof_by_seed"][i]["metrics"]["state_pair_sign_accuracy"]["median"]) <= 1e-12
        and abs(item["landscape_gain_recovery"] - summary["pooled_oof_by_seed"][i]["landscape_gain_recovery"]) <= 1e-12
        for i, item in enumerate(pooled)
    )
    expected_fail = pass_count < thresholds["minimum_pooled_seeds_passing_all_gates"]
    checks["qualification_consistent"] = expected_fail and summary["qualification"] == "QUERY_SINGLE_CENTER_PRETRAIN_OFFLINE_FAIL_NO_OAC"
    qualification = (
        "QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_CONFIRMED_FAIL_NO_OAC"
        if all(checks.values()) and expected_fail
        else "QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_VALIDATION_ERROR"
    )
    validation = {
        "qualification": qualification, "checks": checks,
        "errors": {"actor_knots_max": actor_error, "actor_query_cost_max": actor_cost_error,
                   "critic_log_cost_max": critic_error, "no_anchor_invariance_max": invariance_error},
        "pooled_oof_by_seed": pooled, "offline_seed_pass_count": pass_count,
        "fresh_actor_center_fd_skipped": True,
        "fresh_actor_center_fd_skip_reason": "offline same-state pair-sign gate failed for all pooled seeds",
        "oac_started": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    dump_json(artifact / "validation.json", validation)
    print(json.dumps(validation, indent=2, default=json_default))
    if qualification.endswith("ERROR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
