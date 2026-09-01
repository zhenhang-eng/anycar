#!/usr/bin/env python3
"""Decompose high-speed deterministic DBM J50 into along/cross position terms."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import interpolate_knots


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_STRONG = Path(
    "outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_along_cross_cost_20260830_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--strong-dir", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def metrics(
    names: np.ndarray,
    total: np.ndarray,
    components: dict[str, np.ndarray],
    along_rmse: np.ndarray,
    cross_rmse: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    warm = total[:, 0]
    for column, raw_name in enumerate(names):
        name = str(raw_name)
        item: dict[str, Any] = {
            "total_cost": distribution(total[:, column]),
            "along_rmse_m": distribution(along_rmse[:, column]),
            "cross_rmse_m": distribution(cross_rmse[:, column]),
            "components": {
                key: distribution(value[:, column]) for key, value in components.items()
            },
            "component_sum_share": {
                key: float(value[:, column].sum() / total[:, column].sum())
                for key, value in components.items()
            },
            "position_split": {
                "along_fraction": float(
                    components["position_along"][:, column].sum()
                    / components["position"][:, column].sum()
                ),
                "cross_fraction": float(
                    components["position_cross"][:, column].sum()
                    / components["position"][:, column].sum()
                ),
            },
        }
        if column:
            gain = warm - total[:, column]
            aggregate_gain = float(gain.sum())
            item["warm_relative"] = {
                "aggregate_cost_reduction_fraction": float(gain.sum() / warm.sum()),
                "strict_improvement_fraction": float(np.mean(gain > 1e-5)),
                "gain": distribution(gain),
                "component_gain_sum": {
                    key: float(
                        (components[key][:, 0] - components[key][:, column]).sum()
                    )
                    for key in components
                },
                "component_gain_fraction_of_total": {
                    key: float(
                        (components[key][:, 0] - components[key][:, column]).sum()
                        / aggregate_gain
                    )
                    for key in components
                } if abs(aggregate_gain) > 1e-12 else {},
            }
        result[name] = item
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    replay_path = (args.replay_dir / "replay.npz").resolve()
    strong_path = (args.strong_dir / "oracle.npz").resolve()
    strong_summary_path = (args.strong_dir / "summary.json").resolve()
    strong_validator_path = (args.strong_dir / "validator_report.json").resolve()
    strong_summary = json.loads(strong_summary_path.read_text())
    strong_validator = json.loads(strong_validator_path.read_text())
    if strong_validator["qualification"] != "HIGHSPEED_STRONG_SEARCH_ORACLE_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("strong-search source did not pass independent replay")
    if strong_summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("strong-search source is not train-only")

    with np.load(replay_path, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(strong_path, allow_pickle=False) as loaded:
        strong = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = strong["source_indices"].astype(np.int64)
    if len(rows) != 120 or not np.array_equal(replay["episode_id"][rows], strong["episode_id"]):
        raise AssertionError("strong-search/replay state alignment failed")

    names = np.concatenate((strong["start_names"], np.asarray(["strong_oracle"])))
    centers = np.concatenate(
        (strong["start_centers"], strong["oracle_centers"][:, None]), axis=1
    ).astype(np.float32)
    expected_cost = np.concatenate(
        (strong["start_costs"], strong["oracle_costs"][:, None]), axis=1
    ).astype(np.float32)
    batch, candidates = centers.shape[:2]
    params = TorchMPPIParams(num_samples=64)
    weights = TorchMPPICostWeights()
    backend = TorchDynamicBicycleRolloutBackend()
    device = torch.device(args.device)

    center_t = torch.from_numpy(centers).to(device)
    actions = interpolate_knots(center_t, params.horizon)
    flat_actions = actions.reshape(batch * candidates, params.horizon, 2)
    initial = torch.from_numpy(replay["state_six"][rows]).to(device)
    flat_initial = initial[:, None].expand(-1, candidates, -1).reshape(batch * candidates, 6)
    with torch.no_grad():
        full = backend.rollout_full_state_differentiable(flat_initial, flat_actions)
    trajectory = full[..., [0, 1, 2, 3, 5]].reshape(batch, candidates, params.horizon, 5)
    reference = torch.from_numpy(replay["reference"][rows, 1:]).to(device)
    current_action = torch.from_numpy(replay["current_action"][rows]).to(device)

    error = trajectory[..., :2] - reference[:, None, :, :2]
    reference_yaw = reference[:, None, :, 2]
    tangent = torch.stack((torch.cos(reference_yaw), torch.sin(reference_yaw)), dim=-1)
    normal = torch.stack((-torch.sin(reference_yaw), torch.cos(reference_yaw)), dim=-1)
    along = (error * tangent).sum(dim=-1)
    cross = (error * normal).sum(dim=-1)
    yaw_delta = trajectory[..., 2] - reference[:, None, :, 2]
    yaw_error = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    vx_error = trajectory[..., 3] - reference[:, None, :, 3]
    previous = torch.cat((
        current_action[:, None, None].expand(-1, candidates, 1, -1),
        actions[:, :, :-1],
    ), dim=2)
    rate = actions - previous

    components_t = {
        "position_along": weights.position * along.square().sum(dim=-1),
        "position_cross": weights.position * cross.square().sum(dim=-1),
        "position": weights.position * error.square().sum(dim=-1).sum(dim=-1),
        "yaw": weights.yaw * yaw_error.square().sum(dim=-1),
        "vx": weights.vx * vx_error.square().sum(dim=-1),
        "acceleration_rate": weights.acceleration_rate * rate[..., 0].square().sum(dim=-1),
        "steering_rate": weights.steering_rate * rate[..., 1].square().sum(dim=-1),
    }
    components = {key: value.cpu().numpy().astype(np.float32) for key, value in components_t.items()}
    total = (
        components["position"] + components["yaw"] + components["vx"]
        + components["acceleration_rate"] + components["steering_rate"]
    )
    along_rmse = torch.sqrt(along.square().mean(dim=-1)).cpu().numpy().astype(np.float32)
    cross_rmse = torch.sqrt(cross.square().mean(dim=-1)).cpu().numpy().astype(np.float32)
    identity_error = np.abs(
        components["position"]
        - components["position_along"] - components["position_cross"]
    )
    replay_error = np.abs(total - expected_cost)

    overall = metrics(names, total, components, along_rmse, cross_rmse)
    by_speed: dict[str, Any] = {}
    for speed in sorted(np.unique(strong["nominal_speed_kph"]).tolist()):
        mask = np.isclose(strong["nominal_speed_kph"], speed)
        by_speed[str(int(round(float(speed))))] = metrics(
            names, total[mask], {key: value[mask] for key, value in components.items()},
            along_rmse[mask], cross_rmse[mask],
        )

    output.mkdir(parents=True)
    np.savez_compressed(
        output / "decomposition.npz",
        source_indices=rows,
        episode_id=strong["episode_id"],
        scenario_class=strong["scenario_class"],
        nominal_speed_kph=strong["nominal_speed_kph"],
        actual_vx_mps=strong["actual_vx_mps"],
        proposal_names=names,
        centers=centers,
        total_cost=total,
        expected_cost=expected_cost,
        along_rmse_m=along_rmse,
        cross_rmse_m=cross_rmse,
        **components,
    )
    artifact_path = output / "decomposition.npz"
    summary = {
        "format": "highspeed_along_cross_cost_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "train-only independent step-0 mechanism subset",
            "state_count": int(batch),
            "proposal_count": int(candidates),
            "proposal_names": names.tolist(),
            "horizon": params.horizon,
            "dt": params.dt,
            "association": "fixed time index; tangent/normal from stored reference yaw",
            "nearest_point_or_frenet_projection": False,
            "formal_validation_or_test_opened": False,
        },
        "cost_weights": vars(weights),
        "checks": {
            "position_identity_max_abs_error": float(identity_error.max()),
            "stored_total_cost_max_abs_error": float(replay_error.max()),
            "stored_total_cost_max_relative_error": float(
                np.max(replay_error / np.maximum(np.abs(expected_cost), 1.0))
            ),
        },
        "inputs": {
            "replay": str(replay_path), "replay_sha256": sha256(replay_path),
            "strong_oracle": str(strong_path), "strong_oracle_sha256": sha256(strong_path),
            "strong_summary": str(strong_summary_path),
            "strong_summary_sha256": sha256(strong_summary_path),
            "strong_validator": str(strong_validator_path),
            "strong_validator_sha256": sha256(strong_validator_path),
        },
        "overall": overall,
        "by_nominal_speed_kph": by_speed,
        "qualification": "HIGHSPEED_ALONG_CROSS_DECOMPOSITION_PENDING_INDEPENDENT_VALIDATION",
    }
    summary["artifact_sha256"] = sha256(artifact_path)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "checks": summary["checks"],
        "warm": overall["warm"],
        "strong_oracle": overall["strong_oracle"],
    }, indent=2))


if __name__ == "__main__":
    main()
