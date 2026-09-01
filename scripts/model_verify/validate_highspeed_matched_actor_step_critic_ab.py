#!/usr/bin/env python3
"""Independent artifact and DBM replay validation for matched Actor-step A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from pretrain_highspeed_actor_twin_critic import load_data
from train_highspeed_actor_visited_oac import rollout_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--cost-atol-scale", type=float, default=2e-6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_data(source_root: Path) -> dict[str, np.ndarray]:
    contract = json.loads((source_root / "contract.json").read_text())
    pretrain_root = Path(contract["source_pretrain"])
    summary = json.loads((pretrain_root / "summary.json").read_text())
    source = summary["source"]
    data = load_data(Path(source["replay_dir"]), Path(source["teacher_dir"]))
    indices = np.asarray(summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    return {
        key: value[indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    analysis_path = root / "analysis.json"
    arrays_path = root / "evaluation.npz"
    analysis = json.loads(analysis_path.read_text())
    if analysis["qualification"] != "HIGHSPEED_MATCHED_ACTOR_OUTPUT_STEP_CRITIC_AB_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected analysis qualification")
    if analysis["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("forbidden split created")
    if analysis["contract"]["DBM_role"] != "post-step scalar J50 evaluation only; no analytic gradient":
        raise AssertionError("DBM role mismatch")
    if sha256(arrays_path) != analysis["sources"]["evaluation_sha256"]:
        raise AssertionError("evaluation hash mismatch")
    source_root = Path(
        "outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1"
    ).resolve()
    budget_root = Path(
        "outputs/mppi_proposal/highspeed_search_replay_critic_budget_20260830_v1"
    ).resolve()
    for path, key in (
        (source_root / "summary.json", "source_summary_sha256"),
        (source_root / "validator_report.json", "source_validator_sha256"),
        (budget_root / "summary.json", "budget_summary_sha256"),
        (budget_root / "validator_report.json", "budget_validator_sha256"),
    ):
        if sha256(path) != analysis["sources"][key]:
            raise AssertionError(f"upstream hash mismatch: {path}")
    data = load_shared_data(source_root)
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    count = len(arrays["fold"])
    expected_count = len(analysis["records"]) * 2 * 2 * len(
        analysis["contract"]["target_radii_sigma"]
    )
    if count != expected_count:
        raise AssertionError("evaluation row count mismatch")
    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    max_cost_error = 0.0
    max_base_cost_error = 0.0
    leakage_count = 0
    for index in range(count):
        oof = arrays["oof"][index]
        fold = int(arrays["fold"][index])
        expected_episode = set(data["episode"][oof].tolist())
        source_record = next(
            row for row in analysis["records"]
            if int(row["fold"]) == fold and int(row["seed"]) == int(arrays["seed"][index])
        )
        source_summary = json.loads((source_root / "summary.json").read_text())
        source_row = next(
            row for row in source_summary["records"]
            if int(row["fold"]) == fold and int(row["seed"]) == int(arrays["seed"][index])
        )
        payload = torch.load(Path(source_row["checkpoint"]), map_location="cpu")
        fit_episode = set(data["episode"][np.asarray(payload["fit_indices"], np.int64)].tolist())
        if expected_episode & fit_episode:
            leakage_count += 1
        replay_cost = rollout_bank(
            data, arrays["action"][index][:, None], oof, backend, weights,
            params, device, args.rollout_batch_size,
        )[:, 0]
        base_replay_cost = rollout_bank(
            data, arrays["base_action"][index][:, None], oof, backend, weights,
            params, device, args.rollout_batch_size,
        )[:, 0]
        max_cost_error = max(
            max_cost_error, float(np.max(np.abs(replay_cost - arrays["cost"][index])))
        )
        max_base_cost_error = max(
            max_base_cost_error,
            float(np.max(np.abs(base_replay_cost - arrays["base_cost"][index]))),
        )
        step = next(
            row for row in source_record["steps"]
            if row["arm"] == str(arrays["arm"][index])
            and row["method"] == str(arrays["method"][index])
            and np.isclose(row["target_rms_sigma"], arrays["radius"][index])
        )
        gain = arrays["base_cost"][index].astype(np.float64) - arrays["cost"][index]
        if abs(float(gain.mean()) - float(step["metrics"]["oof"]["gain"]["mean"])) > 1e-9:
            raise AssertionError("stored mean gain mismatch")
        if abs(float(np.quantile(gain, 0.05)) - float(step["metrics"]["oof"]["gain"]["p05"])) > 1e-9:
            raise AssertionError("stored P05 gain mismatch")
    if leakage_count:
        raise AssertionError(f"episode leakage in {leakage_count} rows")
    tolerance = max(1e-3, args.cost_atol_scale * float(np.max(data["anchor_cost"])))
    if max(max_cost_error, max_base_cost_error) > tolerance:
        raise AssertionError("DBM replay tolerance exceeded")
    report = {
        "format": "highspeed_matched_actor_step_critic_ab_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_MATCHED_ACTOR_OUTPUT_STEP_CRITIC_AB_INDEPENDENT_REPLAY_PASS",
        "contract": {
            "evaluation_rows": count, "episode_leakage_count": leakage_count,
            "DBM_role": "post-step scalar J50 replay only",
            "formal_validation_or_test_created": False,
        },
        "checks": {
            "analysis_sha256": sha256(analysis_path),
            "evaluation_sha256": sha256(arrays_path),
            "upstream_hashes_verified": True,
            "candidate_dbm_cost_max_error": max_cost_error,
            "base_dbm_cost_max_error": max_base_cost_error,
            "cost_tolerance": tolerance,
            "stored_gain_statistics_recomputed": True,
        },
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
