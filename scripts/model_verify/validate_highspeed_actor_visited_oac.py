#!/usr/bin/env python3
"""Independently reload and replay train-only high-speed Actor-visited OAC."""

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
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from pretrain_highspeed_actor_twin_critic import load_data, rollout_cost
from train_highspeed_actor_visited_oac import (
    build_inputs,
    local_critic_metrics,
    rollout_bank,
)


DEFAULT_ROOT = Path(
    "outputs/mppi_proposal/highspeed_actor_visited_oac_20260830_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_predict(
    model: DirectNoAnchorGTXActor, inputs: tuple[np.ndarray, ...],
    rows: np.ndarray, device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            local = rows[start : start + 64]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
            output.append(model(*tensors)[1].cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def recovery(cost: np.ndarray, rows: np.ndarray, data: dict[str, np.ndarray]) -> float:
    gain = data["anchor_cost"][rows] - cost
    available = data["anchor_cost"][rows] - data["teacher_cost"][rows]
    return float(np.sum(gain) / np.sum(available))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    contract_path = root / "contract.json"
    summary_path = root / "summary.json"
    contract = json.loads(contract_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_COMPLETE_TRAIN_ONLY":
        raise AssertionError("unexpected OAC qualification")
    if sha256(contract_path) != summary["contract_sha256"]:
        raise AssertionError("contract hash mismatch")
    if contract["formal_validation_or_test_created"] or summary["formal_validation_or_test_created"]:
        raise AssertionError("formal validation/test was unexpectedly opened")
    if contract["warm_in_actor_loss"]:
        raise AssertionError("warm was unexpectedly used by Actor loss")
    if contract["current_input"] != "[vx, yaw_rate, acceleration, steering]; no vy/beta":
        raise AssertionError("unexpected current-state contract")

    pretrain_root = Path(contract["source_pretrain"])
    pretrain_summary_path = pretrain_root / "summary.json"
    pretrain_validator_path = pretrain_root / "validator_report.json"
    if sha256(pretrain_summary_path) != contract["source_pretrain_summary_sha256"]:
        raise AssertionError("source pretrain summary hash mismatch")
    if sha256(pretrain_validator_path) != contract["source_pretrain_validator_sha256"]:
        raise AssertionError("source pretrain validator hash mismatch")
    pretrain_summary = json.loads(pretrain_summary_path.read_text())
    source = pretrain_summary["source"]
    data = load_data(Path(source["replay_dir"]), Path(source["teacher_dir"]))
    source_indices = np.asarray(pretrain_summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    data = {
        key: value[source_indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    records = []
    center_repeat_errors = []
    center_stored_errors = []
    center_cost_errors = []
    probe_cost_errors = []
    replay_cost_errors = []
    recovery_errors = []
    critic_metric_errors = []
    leakage_count = 0

    for expected in summary["records"]:
        checkpoint_path = Path(expected["checkpoint"])
        replay_path = Path(expected["replay"])
        if sha256(checkpoint_path) != expected["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {checkpoint_path}")
        if sha256(replay_path) != expected["replay_sha256"]:
            raise AssertionError(f"replay hash mismatch: {replay_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        if payload["formal_validation_or_test_created"]:
            raise AssertionError("checkpoint opened formal validation/test")
        if int(payload["selected_round"]) != int(expected["selected_round"]):
            raise AssertionError("selected round mismatch")
        fit = np.asarray(payload["fit_indices"], np.int64)
        selection = np.asarray(payload["selection_indices"], np.int64)
        oof = np.asarray(payload["oof_indices"], np.int64)
        split_episodes = [set(data["episode"][rows]) for rows in (fit, selection, oof)]
        local_leakage = sum(len(split_episodes[i] & split_episodes[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
        leakage_count += local_leakage
        if local_leakage:
            raise AssertionError("episode-group leakage")

        inputs = build_inputs(data, payload)
        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
        actor.load_state_dict(payload["actor_selected_state_dict"], strict=True)
        center = actor_predict(actor, inputs, oof, device)
        center_repeat = actor_predict(actor, inputs, oof, device)
        center_repeat_errors.append(float(np.max(np.abs(center - center_repeat))))
        fresh_cost = rollout_cost(
            data, center, oof, backend, weights, params, device, args.batch_size
        )

        with np.load(replay_path, allow_pickle=False) as loaded:
            replay = {name: np.asarray(loaded[name]) for name in loaded.files}
        center_stored_errors.append(float(np.max(np.abs(
            center - replay["selected_oof_center"]
        ))))
        center_cost_errors.append(float(np.max(np.abs(fresh_cost - replay["selected_oof_cost"]))))
        fresh_recovery = recovery(fresh_cost, oof, data)
        expected_recovery = float(expected["selected"]["oof"]["teacher_gain_recovery"])
        recovery_errors.append(abs(fresh_recovery - expected_recovery))

        probe_cost = rollout_bank(
            data, replay["oof_probe_bank"], oof, backend, weights, params,
            device, args.batch_size,
        )
        probe_cost_errors.append(float(np.max(np.abs(probe_cost - replay["oof_probe_cost"]))))
        critics = []
        training = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            critics.append(critic)
            training.append(payload[f"critic{twin}_training"])
        fresh_critic = local_critic_metrics(
            tuple(critics), tuple(training), inputs, oof,
            replay["oof_probe_bank"], probe_cost, device,
        )
        for key in (
            "centered_log_cost_pearson", "center_relative_sign_accuracy",
            "bank_gain_recovery", "selected_beats_center_fraction",
        ):
            critic_metric_errors.append(abs(
                float(fresh_critic[key]) - float(expected["critic_oof_local_probe"][key])
            ))

        # Replay initial, first online, and last online candidates on three fit states.
        state_local = np.asarray((0, len(fit) // 2, len(fit) - 1), np.int64)
        candidate_local = np.unique(np.asarray(
            (0, 128, 129, replay["actions"].shape[1] - 1), np.int64
        ))
        sample_actions = replay["actions"][state_local[:, None], candidate_local]
        sample_rows = np.repeat(fit[state_local], len(candidate_local))
        sample_cost = rollout_cost(
            data, sample_actions.reshape(-1, 8, 2), sample_rows,
            backend, weights, params, device, args.batch_size,
        ).reshape(len(state_local), len(candidate_local))
        replay_cost_errors.append(float(np.max(np.abs(
            sample_cost - replay["costs"][state_local[:, None], candidate_local]
        ))))
        records.append({
            "fold": int(payload["fold"]), "seed": int(payload["seed"]),
            "selected_round": int(payload["selected_round"]),
            "selected_oof_teacher_gain_recovery": fresh_recovery,
            "critic_center_relative_sign_accuracy": fresh_critic["center_relative_sign_accuracy"],
            "episode_disjoint": True,
        })

    maxima = {
        "actor_center_repeat_max_abs_error": float(max(center_repeat_errors)),
        "actor_center_vs_stored_max_abs_error": float(max(center_stored_errors)),
        "actor_center_max_abs_error": float(max(
            max(center_repeat_errors), max(center_stored_errors)
        )),
        "actor_center_dbm_cost_max_abs_error": float(max(center_cost_errors)),
        "oof_probe_dbm_cost_max_abs_error": float(max(probe_cost_errors)),
        "sampled_replay_dbm_cost_max_abs_error": float(max(replay_cost_errors)),
        "actor_recovery_max_abs_error_vs_summary": float(max(recovery_errors)),
        "critic_metric_max_abs_error_vs_summary": float(max(critic_metric_errors)),
    }
    print(json.dumps({"replay_maxima": maxima}, indent=2), flush=True)
    # CUDA convolution/attention kernels can differ by tens of micro action
    # units across a long multi-checkpoint replay.  High-speed J50 amplifies
    # that tiny center delta, so retain both raw center/cost errors and use a
    # tight action-space gate rather than demanding bitwise CUDA equivalence.
    center_tolerance = 1e-4
    if maxima["actor_center_max_abs_error"] > center_tolerance:
        raise AssertionError("Actor reload/determinism mismatch")
    dbm_tolerance = max(1e-3, 2e-6 * float(data["anchor_cost"].max()))
    actor_reload_cost_tolerance = max(
        dbm_tolerance, 1e-4 * float(data["anchor_cost"].max())
    )
    if max(
        maxima["oof_probe_dbm_cost_max_abs_error"],
        maxima["sampled_replay_dbm_cost_max_abs_error"],
    ) > dbm_tolerance:
        raise AssertionError("saved-action DBM replay mismatch")
    if maxima["actor_center_dbm_cost_max_abs_error"] > actor_reload_cost_tolerance:
        raise AssertionError("reloaded-Actor DBM cost mismatch")
    recovery_tolerance = 1e-4
    if maxima["actor_recovery_max_abs_error_vs_summary"] > recovery_tolerance:
        raise AssertionError("Actor summary mismatch")
    if maxima["critic_metric_max_abs_error_vs_summary"] > 1e-6:
        raise AssertionError("Critic summary mismatch")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS",
        "summary_sha256": sha256(summary_path),
        "contract_sha256": sha256(contract_path),
        "checkpoint_count": len(records),
        "actor_center_absolute_tolerance": center_tolerance,
        "dbm_absolute_tolerance": dbm_tolerance,
        "actor_reload_dbm_cost_absolute_tolerance": actor_reload_cost_tolerance,
        "actor_recovery_absolute_tolerance": recovery_tolerance,
        **maxima,
        "episode_group_leakage_count": leakage_count,
        "warm_in_actor_loss": False,
        "formal_validation_or_test_created": False,
        "records": records,
    }
    report_path = root / "validator_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
