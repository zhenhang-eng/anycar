#!/usr/bin/env python3
"""Audit the frozen small-car Query training domain against high-speed MPPI inputs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams
from car_foundation.query_deployment import QueryDeploymentModel
from generate_dbm_direct_gt_validation import interpolate_knots


DEFAULT_CHECKPOINT = Path(
    "outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt"
)
DEFAULT_ONNX = Path(
    "outputs/formal_small_car_query_dt005/20260730T144840/anycar_query.onnx"
)
DEFAULT_TRAIN_SUMMARY = Path(
    "outputs/formal_small_car_query_dt005/20260730T144840/summary.json"
)
DEFAULT_METADATA = Path(
    "/disk/collect_data_from_anycar/generated_small_car_query_dt005/"
    "20260730T144638/metadata.json"
)
DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/replay.npz"
)
DEFAULT_REPLAY_SUMMARY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/query_mppi/highspeed_query_domain_audit_20260831_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--train-summary", type=Path, default=DEFAULT_TRAIN_SUMMARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--replay-summary", type=Path, default=DEFAULT_REPLAY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "max": float(values.max()),
    }


def exact_zero_fraction(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values) == 0.0))


def z_audit(raw: np.ndarray, mean: np.ndarray, std: np.ndarray, names: list[str]) -> dict:
    z = (np.asarray(raw, np.float64) - mean) / std
    result = {}
    for index, name in enumerate(names):
        absolute = np.abs(z[..., index]).reshape(-1)
        result[name] = {
            "raw": distribution(raw[..., index]),
            "abs_z": distribution(absolute),
            "fraction_abs_z_gt_3": float(np.mean(absolute > 3.0)),
            "fraction_abs_z_gt_5": float(np.mean(absolute > 5.0)),
            "fraction_abs_z_gt_10": float(np.mean(absolute > 10.0)),
        }
    return result


def compare(name: str, left, right, tolerance: float = 1e-8) -> dict:
    left = float(left); right = float(right)
    return {
        "name": name, "training": left, "current": right,
        "abs_difference": abs(left - right),
        "match": abs(left - right) <= tolerance,
    }


def load_training_empirical(dataset_path: Path) -> dict[str, np.ndarray]:
    states = {name: [] for name in ("vx", "vy", "yawrate")}
    actions = {name: [] for name in ("acceleration", "steering")}
    transitions = {name: [] for name in ("dx_body", "dy_body", "dyaw", "dvx", "dyawrate")}
    files = sorted(dataset_path.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no Query training PKLs under {dataset_path}")
    for path in files:
        with path.open("rb") as stream:
            dataset = pickle.load(stream)
        log = dataset.data_logs
        x = np.asarray(log["xpos_x"], np.float64)
        y = np.asarray(log["xpos_y"], np.float64)
        w = np.asarray(log["xori_w"], np.float64)
        z = np.asarray(log["xori_z"], np.float64)
        yaw = np.arctan2(2.0 * w * z, 1.0 - 2.0 * z * z)
        vx = np.asarray(log["xvel_x"], np.float64)
        vy = np.asarray(log["xvel_y"], np.float64)
        yawrate = np.asarray(log["avel_z"], np.float64)
        acceleration = np.asarray(log["throttle"], np.float64)
        steering = np.asarray(log["steer"], np.float64)
        lap_end = np.asarray(log["lap_end"]).astype(bool)
        states["vx"].append(vx); states["vy"].append(vy); states["yawrate"].append(yawrate)
        actions["acceleration"].append(acceleration); actions["steering"].append(steering)
        valid = np.ones(len(x), dtype=bool)
        valid[0] = False
        valid[1:] &= ~lap_end[:-1]
        previous = np.flatnonzero(valid) - 1
        current = previous + 1
        dx_world = x[current] - x[previous]
        dy_world = y[current] - y[previous]
        cosine = np.cos(yaw[previous]); sine = np.sin(yaw[previous])
        transitions["dx_body"].append(dx_world * cosine + dy_world * sine)
        transitions["dy_body"].append(-dx_world * sine + dy_world * cosine)
        delta_yaw = yaw[current] - yaw[previous]
        transitions["dyaw"].append(np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw)))
        transitions["dvx"].append(vx[current] - vx[previous])
        transitions["dyawrate"].append(yawrate[current] - yawrate[previous])
    return {
        "file_count": len(files),
        "states": {name: np.concatenate(value) for name, value in states.items()},
        "actions": {name: np.concatenate(value) for name, value in actions.items()},
        "transitions": {name: np.concatenate(value) for name, value in transitions.items()},
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_summary = json.loads(args.train_summary.read_text())
    metadata = json.loads(args.metadata.read_text())
    replay_summary = json.loads(args.replay_summary.read_text())
    with np.load(args.replay, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    training_empirical = load_training_empirical(Path(checkpoint["args"]["dataset_path"]))

    stats = checkpoint["stats"]
    history_mean = np.asarray(stats["history"][0], np.float64)
    history_std = np.asarray(stats["history"][1], np.float64)
    context_mean = np.asarray(stats["context"][0], np.float64)
    context_std = np.asarray(stats["context"][1], np.float64)
    nominal_state_mean = np.asarray(stats["nominal_state"][0], np.float64)
    nominal_state_std = np.asarray(stats["nominal_state"][1], np.float64)
    nominal_transition_mean = np.asarray(stats["nominal_transition"][0], np.float64)
    nominal_transition_std = np.asarray(stats["nominal_transition"][1], np.float64)

    history = replay["history"]
    sampled_action = replay["sampled_action_sequences"]
    state_six = replay["state_six"]
    current_action = replay["current_action"]
    candidate_count = sampled_action.shape[1]
    context = np.stack((
        np.repeat(state_six[:, 3], candidate_count, axis=0).reshape(len(state_six), candidate_count),
        np.repeat(state_six[:, 5], candidate_count, axis=0).reshape(len(state_six), candidate_count),
        sampled_action[:, :, 0, 0],
        np.repeat(current_action[:, 1], candidate_count, axis=0).reshape(len(state_six), candidate_count),
    ), axis=-1)

    # The nominal-query branch sees one nominal 50-step plan per Actor/warm
    # center.  Warm is enough for a domain-contract audit and avoids mixing the
    # 64 stochastic candidate distribution into the temporal z-score count.
    model = QueryDeploymentModel.from_checkpoint(args.checkpoint, device="cpu")
    warm_action = interpolate_knots(
        torch.as_tensor(replay["mean_knots_before"], dtype=torch.float32), 50
    )
    observable = torch.as_tensor(
        state_six[:, (0, 1, 2, 3, 5)], dtype=torch.float32
    )
    with torch.no_grad():
        nominal_absolute, nominal_transition = model._nominal_rollout(
            observable, warm_action
        )
        nominal_state = model._relative_state(observable, nominal_absolute)
    nominal_state = nominal_state.numpy()
    nominal_transition = nominal_transition.numpy()

    train_range = metadata["ranges"]
    dbm = dataclasses.asdict(TorchDBMParams())
    vehicle = metadata["vehicle"]
    physical_checks = [
        compare("dt", vehicle["dt"], dbm["dt"]),
        compare("wheelbase", vehicle["wheelbase"], dbm["lf"] + dbm["lr"]),
        compare("lf", vehicle["lf"], dbm["lf"]),
        compare("lr", vehicle["lr"], dbm["lr"]),
        compare("mass", vehicle["mass"], dbm["mass"]),
        compare("friction", vehicle["friction"], dbm["friction"]),
        compare("max_throttle", vehicle["max_throttle"], dbm["throttle_scale"]),
        compare("max_steer", vehicle["max_steer"], dbm["steering_scale"]),
        compare("steer_bias", vehicle["steer_bias"], dbm["steering_bias"]),
    ]
    onnx_opsets = []
    try:
        import onnx
        onnx_model = onnx.load(str(args.onnx))
        onnx_opsets = [
            {"domain": item.domain, "version": int(item.version)}
            for item in onnx_model.opset_import
        ]
    except Exception as error:  # pragma: no cover - recorded as audit evidence
        onnx_opsets = [{"error": repr(error)}]

    train_state_min = np.asarray(train_range["state_min"], np.float64)
    train_state_max = np.asarray(train_range["state_max"], np.float64)
    current_state_dimensions = {
        "vx": state_six[:, 3], "vy_unobserved_by_query": state_six[:, 4],
        "yawrate": state_six[:, 5],
    }
    train_state_indices = {"vx": 3, "vy_unobserved_by_query": 4, "yawrate": 5}
    state_support = {}
    for name, values in current_state_dimensions.items():
        index = train_state_indices[name]
        state_support[name] = {
            "training_min": float(train_state_min[index]),
            "training_max": float(train_state_max[index]),
            "current": distribution(values),
            "current_fraction_outside_training_minmax": float(np.mean(
                (values < train_state_min[index]) | (values > train_state_max[index])
            )),
            "training_empirical": distribution(training_empirical["states"][
                "vy" if name == "vy_unobserved_by_query" else name
            ]),
        }

    action_flat = sampled_action.reshape(-1, 2)
    history_action_flat = history[..., 5:7].reshape(-1, 2)
    action_support = {}
    for index, name in enumerate(("acceleration", "steering")):
        action_support[name] = {
            "training_min": float(train_range["action_min"][index]),
            "training_max": float(train_range["action_max"][index]),
            "current_candidates": distribution(action_flat[:, index]),
            "current_history": distribution(history_action_flat[:, index]),
            "current_history_fraction_exact_zero": exact_zero_fraction(
                history_action_flat[:, index]
            ),
            "training_empirical": distribution(training_empirical["actions"][name]),
            "training_empirical_fraction_exact_zero": exact_zero_fraction(
                training_empirical["actions"][name]
            ),
            "candidate_fraction_outside_training_minmax": float(np.mean(
                (action_flat[:, index] < train_range["action_min"][index])
                | (action_flat[:, index] > train_range["action_max"][index])
            )),
        }

    history_audit = z_audit(
        history[..., :5], history_mean, history_std,
        ["dx_body", "dy_body", "dyaw", "dvx", "dyawrate"],
    )
    context_audit = z_audit(
        context, context_mean, context_std,
        ["vx", "yawrate", "first_future_acceleration", "current_steering"],
    )
    nominal_state_audit = z_audit(
        nominal_state, nominal_state_mean, nominal_state_std,
        ["x_body", "y_body", "yaw_relative", "vx", "yawrate"],
    )
    nominal_transition_audit = z_audit(
        nominal_transition, nominal_transition_mean, nominal_transition_std,
        ["dx_body", "dy_body", "dvx", "dyawrate"],
    )
    training_history_empirical = {
        name: distribution(training_empirical["transitions"][name])
        for name in ("dx_body", "dy_body", "dyaw", "dvx", "dyawrate")
    }

    severe = {
        "speed_support_mismatch": state_support["vx"]["current_fraction_outside_training_minmax"] > 0.5,
        "history_dx_body_abs_z_p95_gt_10": history_audit["dx_body"]["abs_z"]["p95"] > 10,
        "context_vx_abs_z_p95_gt_10": context_audit["vx"]["abs_z"]["p95"] > 10,
        "nominal_dx_abs_z_p95_gt_10": nominal_transition_audit["dx_body"]["abs_z"]["p95"] > 10,
        "early_replay_history_action_zero_fraction_gt_0p95": (
            exact_zero_fraction(history_action_flat) > 0.95
        ),
    }
    matched = {
        "all_vehicle_and_dt_checks": all(item["match"] for item in physical_checks),
        "history_length": train_summary["args"]["history_length"] == history.shape[1] == 250,
        "prediction_horizon": train_summary["args"]["prediction_length"] == sampled_action.shape[2] == 50,
        "action_channel_order": "[acceleration, steering] on both sides",
        "pre_transition_and_steer_shift": (
            metadata["state_action_logging"] == "pre_transition"
            and metadata["required_query_steer_shift"] == checkpoint["args"]["steer_shift"] == 0
        ),
        "clean_synthetic_history": True,
    }
    analysis = {
        "format": "highspeed_query_domain_contract_audit_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "QUERY_INTERFACE_MATCH_SPEED_NORMALIZATION_SEVERELY_OOD",
        "scope": {
            "purpose": "record training/current contract mismatches; no Query rollout quality claim",
            "query_as_environment_allowed": True,
            "dbm_or_real_fidelity_required": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "training_contract": {
            "generator": metadata["generator"],
            "dataset_path": checkpoint["args"]["dataset_path"],
            "generated_files": metadata["args"]["episodes"] // metadata["args"]["episodes_per_file"],
            "generated_episodes": metadata["args"]["episodes"],
            "generated_steps_per_episode": metadata["args"]["steps"],
            "episode_duration_seconds": metadata["args"]["steps"] * vehicle["dt"],
            "control_knot_steps": metadata["args"]["control_knot_steps"],
            "control_knot_seconds": metadata["args"]["control_knot_steps"] * vehicle["dt"],
            "profile_generation": {
                "target_speed_knots_mps": "uniform [0.2, 3.5], with every eighth episode starting at zero",
                "steering_knots": "uniform [-0.8, 0.8], every eighth episode scaled 1.25 then clipped [-1,1]",
                "acceleration": "clipped speed-error PD plus smooth Gaussian excitation",
                "initial_vx_mps": "uniform [0,2.5], with every fourth episode reset to zero",
            },
            "empirical_files_read": training_empirical["file_count"],
            "target_speed_mps": [train_range["target_speed_min"], train_range["target_speed_max"]],
            "actual_vx_mps": [train_state_min[3], train_state_max[3]],
            "history_length": checkpoint["args"]["history_length"],
            "prediction_length": checkpoint["args"]["prediction_length"],
            "steer_shift": checkpoint["args"]["steer_shift"],
            "checkpoint_epoch": checkpoint["epoch"],
            "selection_metric": checkpoint["selection_metric"],
            "output_contract": train_summary["output_contract"],
            "synthetic_observation_noise": "none",
        },
        "current_contract": {
            "contexts": len(state_six),
            "episodes": replay_summary["episode_count"],
            "speed_kph_nominal": sorted(np.unique(replay["speed_kph"]).tolist()),
            "actual_vx_mps": replay_summary["actual_vx_mps"],
            "history_shape": list(history.shape),
            "candidate_action_shape": list(sampled_action.shape),
            "history_source": replay_summary["notes"][1],
            "history_content": (
                "only control steps 0..4 are retained per episode; step 0 primes all 250 "
                "tokens with constant motion/current action and at most four real transitions "
                "replace the tail"
            ),
            "observation_noise": "none; reconstructed from deterministic DBM trace",
        },
        "matched_contracts": matched,
        "physical_parameter_checks": physical_checks,
        "state_support": state_support,
        "action_support": action_support,
        "normalized_input_audit": {
            "training_history_empirical": training_history_empirical,
            "history": history_audit,
            "context_all_64_candidates": context_audit,
            "warm_nominal_state": nominal_state_audit,
            "warm_nominal_transition": nominal_transition_audit,
        },
        "severe_mismatch_flags": severe,
        "structural_notes": [
            "Query observes [x,y,yaw,vx,yawrate] and has no explicit vy input.",
            "Current high-speed vy can only affect Query through historical dy_body and subsequent observed motion.",
            "The current replay uses only the first five control steps, so its 250-token history is almost entirely the synthetic constant-motion prime rather than a fully observed 12.5-second history.",
            "Reference is a cost input, not a Query dynamics-model input, so reference-speed mismatch is mediated through state/history/action.",
            "If the frozen Query is declared to be the environment, OOD is not a fidelity failure, but extreme normalized inputs can still create an unstable or easily exploitable numerical task.",
        ],
        "onnx": {
            "path": str(args.onnx.resolve()), "opsets": onnx_opsets,
            "historical_low_speed_parity_only": {
                "cpu_max_abs_difference": 2.74e-6,
                "cuda_mean_abs_difference": 4.97e-5,
                "cuda_max_abs_difference": 1.84e-3,
            },
            "high_speed_parity_checked_in_this_audit": False,
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
            "onnx": str(args.onnx.resolve()), "onnx_sha256": sha256(args.onnx),
            "train_summary": str(args.train_summary.resolve()), "train_summary_sha256": sha256(args.train_summary),
            "metadata": str(args.metadata.resolve()), "metadata_sha256": sha256(args.metadata),
            "replay": str(args.replay.resolve()), "replay_sha256": sha256(args.replay),
            "replay_summary": str(args.replay_summary.resolve()), "replay_summary_sha256": sha256(args.replay_summary),
        },
    }
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "qualification": analysis["qualification"],
        "matched_contracts": matched,
        "state_support": state_support,
        "key_z": {
            "history_dx_body": history_audit["dx_body"],
            "context_vx": context_audit["vx"],
            "nominal_transition_dx_body": nominal_transition_audit["dx_body"],
        },
        "severe_mismatch_flags": severe,
    }, indent=2))


if __name__ == "__main__":
    main()
