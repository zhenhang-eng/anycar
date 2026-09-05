#!/usr/bin/env python3
"""Collect causal closed-loop MPPI data with frozen Query dynamics.

Real train-only Query contexts provide only the initial history, state, and
current action.  A procedural desired road supplies the reference.  The frozen
Query model is both the MPPI rollout environment and the deterministic plant:
the first state of the optimized action-sequence rollout becomes the next
closed-loop state.  No recorded future state or future action is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics", "car_planner"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    FIXED_HADAMARD_BANK_VERSION,
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "outputs/formal_real_finetune_query_baseline_split/"
    "20260728T143256/query_best.pt"
)
DEFAULT_SEED_REPLAY = REPO_ROOT / (
    "outputs/query_mppi/real_query_train_distribution_replay_20260901_v1/"
    "replay.npz"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "query_expected_road_train_20260901_v1"
)
SPEED_BINS_KPH = (40, 55, 70, 85, 100)
ROAD_VARIANTS = (
    {
        "name": "mild_left_nominal",
        "kind": "circle",
        "direction": 1,
        "target_yawrate_abs_rps": 0.012,
        "lateral_offset_m": 0.0,
        "heading_error_rad": 0.0,
    },
    {
        "name": "moderate_right_nominal",
        "kind": "circle",
        "direction": -1,
        "target_yawrate_abs_rps": 0.030,
        "lateral_offset_m": 0.0,
        "heading_error_rad": 0.0,
    },
    {
        "name": "varying_left_recovery",
        "kind": "oval",
        "direction": 1,
        "target_yawrate_abs_rps": 0.040,
        "lateral_offset_m": 0.75,
        "heading_error_rad": 0.03,
    },
    {
        "name": "varying_right_recovery",
        "kind": "oval",
        "direction": -1,
        "target_yawrate_abs_rps": 0.040,
        "lateral_offset_m": -0.75,
        "heading_error_rad": -0.03,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed-replay", type=Path, default=DEFAULT_SEED_REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--episodes-per-speed", type=int, default=4)
    parser.add_argument(
        "--repeats-per-cell",
        type=int,
        default=1,
        help="independent episodes for every selected speed/road cell",
    )
    parser.add_argument(
        "--road-variation-fraction",
        type=float,
        default=0.0,
        help=(
            "maximum deterministic fractional variation of curvature and recovery "
            "initialization across cell repeats"
        ),
    )
    parser.add_argument(
        "--variant-indices",
        type=str,
        default="",
        help="comma-separated road variants for focused pilots",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--episode-indices",
        type=str,
        default="",
        help=(
            "comma-separated full-plan episode indices for focused diagnostics; "
            "selection happens after the complete seed plan is constructed"
        ),
    )
    parser.add_argument("--burn-in-steps", type=int, default=250)
    parser.add_argument("--snapshots-per-episode", type=int, default=30)
    parser.add_argument("--snapshot-stride", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument(
        "--sampling-mode",
        choices=("gaussian", "fixed_hadamard_64"),
        default="gaussian",
    )
    parser.add_argument("--seed", type=int, default=9127)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def wrapped(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def selected_variant_indices(args: argparse.Namespace) -> list[int]:
    if args.variant_indices.strip():
        values = [int(value) for value in args.variant_indices.split(",")]
    else:
        values = list(range(args.episodes_per_speed))
    if not values or len(values) != len(set(values)):
        raise ValueError("road variant indices must be nonempty and unique")
    if min(values) < 0 or max(values) >= len(ROAD_VARIANTS):
        raise ValueError(f"road variant indices must be in [0,{len(ROAD_VARIANTS)-1}]")
    return values


def effective_variant(
    variant_index: int,
    speed_kph: int,
    repeat_index: int = 0,
    repeats_per_cell: int = 1,
    road_variation_fraction: float = 0.0,
) -> dict:
    """Use gentler right turns/recovery at the low-speed edge of the domain."""
    spec = dict(ROAD_VARIANTS[variant_index])
    fraction = float(np.clip((speed_kph - 40.0) / 60.0, 0.0, 1.0))
    if int(spec["direction"]) < 0 and speed_kph == 40:
        # Frozen Query is not rolling-one-step stable on 40-kph right turns:
        # its single-shot J50 prediction and history-updated closed loop take
        # opposite yaw branches.  Preserve those pilots as a pressure set and
        # keep the balanced train collection inside the measured stable domain.
        spec["name"] = f"{spec['name']}_40kph_left_substitute"
        spec["direction"] = 1
        spec["target_yawrate_abs_rps"] = 0.025 if variant_index == 1 else 0.030
        # Recovery offsets are signed with the turn direction.  A left-turn
        # substitute must also mirror the original right-turn initialization;
        # retaining negative offset/heading while flipping only curvature makes
        # the recovery target geometrically contradictory.
        spec["lateral_offset_m"] = -float(spec["lateral_offset_m"])
        spec["heading_error_rad"] = -float(spec["heading_error_rad"])
        spec["low_speed_right_turn_substituted"] = True
    elif int(spec["direction"]) < 0:
        maximum = float(spec["target_yawrate_abs_rps"])
        spec["target_yawrate_abs_rps"] = 0.012 + fraction * (maximum - 0.012)
    if "recovery" in str(spec["name"]):
        scale = 0.5 + 0.5 * fraction
        spec["lateral_offset_m"] = float(spec["lateral_offset_m"]) * scale
        spec["heading_error_rad"] = float(spec["heading_error_rad"]) * scale
    speed_index = SPEED_BINS_KPH.index(speed_kph)
    if repeats_per_cell > 1:
        variation_slot = (
            repeat_index + speed_index + 2 * variant_index
        ) % repeats_per_cell
        normalized_variation = (
            2.0 * variation_slot / float(repeats_per_cell - 1) - 1.0
        )
    else:
        variation_slot = 0
        normalized_variation = 0.0
    # The frozen Query's 40-kph yaw branch is especially narrow.  Independent
    # histories/phases are still varied there, but curvature is frozen to the
    # already-qualified value instead of turning a coverage repeat into an
    # unstable road-pressure test.
    applied_variation_fraction = (
        0.0 if speed_kph == 40 else road_variation_fraction
    )
    variation_scale = 1.0 + applied_variation_fraction * normalized_variation
    spec["target_yawrate_abs_rps"] = (
        float(spec["target_yawrate_abs_rps"]) * variation_scale
    )
    if "recovery" in str(spec["name"]):
        recovery_scale = 1.0 + 0.5 * applied_variation_fraction * normalized_variation
        spec["lateral_offset_m"] = float(spec["lateral_offset_m"]) * recovery_scale
        spec["heading_error_rad"] = float(spec["heading_error_rad"]) * recovery_scale
    spec["repeat_index"] = repeat_index
    spec["road_variation_slot"] = variation_slot
    spec["road_variation_normalized"] = normalized_variation
    spec["road_variation_scale"] = variation_scale
    spec["applied_road_variation_fraction"] = applied_variation_fraction
    spec["base_variant_index"] = variant_index
    return spec


def reference_ego(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    result = np.asarray(reference, np.float32).copy()
    delta = result[:, :2] - state[None, :2]
    cosine = math.cos(float(state[2]))
    sine = math.sin(float(state[2]))
    result[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
    result[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
    result[:, 2] = wrapped(result[:, 2] - float(state[2]))
    return result


def append_history(
    history: np.ndarray,
    state: np.ndarray,
    next_state: np.ndarray,
    executed_action: np.ndarray,
) -> np.ndarray:
    dx_world = float(next_state[0] - state[0])
    dy_world = float(next_state[1] - state[1])
    cosine = math.cos(float(state[2]))
    sine = math.sin(float(state[2]))
    token = np.asarray(
        (
            dx_world * cosine + dy_world * sine,
            -dx_world * sine + dy_world * cosine,
            wrapped(float(next_state[2] - state[2])),
            float(next_state[3] - state[3]),
            float(next_state[4] - state[4]),
            float(executed_action[0]),
            float(executed_action[1]),
        ),
        dtype=np.float32,
    )
    return np.concatenate((history[1:], token[None]), axis=0)


class PeriodicArcLengthRoad:
    """Periodic centerline with true metre-valued arc-length sampling."""

    def __init__(self, waypoints: np.ndarray):
        points = np.asarray(waypoints, np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100:
            raise ValueError("waypoints must have shape [N,2] with N>=100")
        closed = np.concatenate((points, points[:1]), axis=0)
        segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        if not np.all(segment > 0):
            raise ValueError("road contains repeated adjacent waypoints")
        self.waypoints = points
        self.s_nodes = np.concatenate(([0.0], np.cumsum(segment)))
        self.total_length = float(self.s_nodes[-1])
        self._x = CubicSpline(self.s_nodes, closed[:, 0], bc_type="periodic")
        self._y = CubicSpline(self.s_nodes, closed[:, 1], bc_type="periodic")
        self._tree = cKDTree(points)

    def sample(self, s: np.ndarray | float) -> np.ndarray:
        value = np.mod(np.asarray(s, np.float64), self.total_length)
        x = self._x(value)
        y = self._y(value)
        yaw = np.arctan2(self._y(value, 1), self._x(value, 1))
        return np.stack((x, y, yaw), axis=-1)

    def project_s(self, position: np.ndarray) -> float:
        _, nearest = self._tree.query(np.asarray(position, np.float64))
        best_distance = float("inf")
        best_s = float(self.s_nodes[int(nearest)])
        count = len(self.waypoints)
        for begin_index in ((int(nearest) - 1) % count, int(nearest)):
            end_index = (begin_index + 1) % count
            begin = self.waypoints[begin_index]
            end = self.waypoints[end_index]
            delta = end - begin
            alpha = float(
                np.clip(
                    np.dot(np.asarray(position) - begin, delta)
                    / np.dot(delta, delta),
                    0.0,
                    1.0,
                )
            )
            projected = begin + alpha * delta
            distance = float(np.linalg.norm(np.asarray(position) - projected))
            if distance < best_distance:
                best_distance = distance
                segment_s = float(self.s_nodes[begin_index])
                if begin_index == count - 1:
                    segment_s = float(self.s_nodes[-2])
                best_s = segment_s + alpha * float(np.linalg.norm(delta))
        return best_s % self.total_length

    def reference(
        self, state: np.ndarray, speed_mps: float, dt: float, count: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        s0 = self.project_s(state[:2])
        s = s0 + np.arange(count, dtype=np.float64) * speed_mps * dt
        pose = self.sample(s)
        result = np.concatenate(
            (pose, np.full((count, 1), speed_mps, np.float64)), axis=1
        ).astype(np.float32)
        tangent = pose[0, 2]
        delta = np.asarray(state[:2], np.float64) - pose[0, :2]
        lateral = -math.sin(tangent) * delta[0] + math.cos(tangent) * delta[1]
        return result, {
            "s_m": float(s0),
            "lateral_m": float(lateral),
            "heading_error_rad": float(wrapped(float(state[2] - tangent))),
        }


def make_waypoints(spec: dict, speed_mps: float) -> tuple[np.ndarray, dict]:
    radius = max(150.0, speed_mps / float(spec["target_yawrate_abs_rps"]))
    waypoint_count = max(4096, int(math.ceil(2.0 * math.pi * 1.10 * radius / 0.5)))
    phase = np.linspace(0.0, 2.0 * np.pi, waypoint_count, endpoint=False)
    if int(spec["direction"]) < 0:
        phase = phase[::-1]
    if spec["kind"] == "circle":
        x_radius = radius
        y_radius = radius
    elif spec["kind"] == "oval":
        x_radius = 1.10 * radius
        y_radius = 0.90 * radius
    else:
        raise ValueError(f"unknown road kind {spec['kind']}")
    waypoints = np.stack(
        (x_radius * np.cos(phase), y_radius * np.sin(phase)), axis=-1
    ).astype(np.float64)
    geometry = {
        **spec,
        "base_radius_m": float(radius),
        "x_radius_m": float(x_radius),
        "y_radius_m": float(y_radius),
        "waypoint_count": int(len(waypoints)),
    }
    return waypoints, geometry


def initial_state_on_road(
    road: PeriodicArcLengthRoad,
    seed_state: np.ndarray,
    spec: dict,
    start_fraction: float,
) -> tuple[np.ndarray, float]:
    total_length = road.total_length
    start_s = float(start_fraction % 1.0) * total_length
    reference_pose = road.sample(start_s)
    reference_yaw = float(reference_pose[2])
    lateral = float(spec["lateral_offset_m"])
    x = float(reference_pose[0] - math.sin(reference_yaw) * lateral)
    y = float(reference_pose[1] + math.cos(reference_yaw) * lateral)
    state = np.asarray(
        (
            x,
            y,
            reference_yaw + float(spec["heading_error_rad"]),
            seed_state[3],
            seed_state[4],
        ),
        dtype=np.float32,
    )
    return state, start_s


def raw_features(
    trajectories: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    current_action: np.ndarray,
) -> dict[str, np.ndarray]:
    target = reference[1:] if len(reference) == 51 else reference
    yaw_delta = wrapped(trajectories[..., 2] - target[None, :, 2])
    previous = np.concatenate(
        (
            np.broadcast_to(current_action, (len(actions), 1, 2)),
            actions[:, :-1],
        ),
        axis=1,
    )
    return {
        "feature_position_error_sq": np.square(
            trajectories[..., :2] - target[None, :, :2]
        ).sum(axis=-1),
        "feature_yaw_error_sq": np.square(yaw_delta),
        "feature_vx_error_sq": np.square(
            trajectories[..., 3] - target[None, :, 3]
        ),
        "feature_action_rate_sq": np.square(actions - previous),
    }


def choose_seed_rows(seed_data: np.lib.npyio.NpzFile, args: argparse.Namespace):
    rows = []
    for variant_index in selected_variant_indices(args):
        for speed in SPEED_BINS_KPH:
            all_eligible = np.flatnonzero(seed_data["speed_bin_kph"] == speed)
            required_per_speed = (
                len(selected_variant_indices(args)) * args.repeats_per_cell
            )
            if len(all_eligible) < required_per_speed:
                raise ValueError(f"not enough seed rows for {speed} kph")
            for repeat_index in range(args.repeats_per_cell):
                used = {
                    item["seed_row"] for item in rows if item["speed_kph"] == speed
                }
                eligible = np.asarray(
                    [index for index in all_eligible if int(index) not in used],
                    dtype=int,
                )
                variant = effective_variant(
                    variant_index,
                    speed,
                    repeat_index,
                    args.repeats_per_cell,
                    args.road_variation_fraction,
                )
                desired_yawrate = (
                    float(variant["direction"])
                    * float(variant["target_yawrate_abs_rps"])
                )
                desired_source = (
                    "real" if speed < 100 and variant_index % 2 == 0 else "pretrain"
                )
                source_match = np.asarray(
                    [
                        index
                        for index in eligible
                        if str(seed_data["source"][index]) == desired_source
                    ],
                    dtype=int,
                )
                if len(source_match):
                    eligible = source_match
                yawrate_error = np.abs(
                    seed_data["state_six"][eligible, 5] - desired_yawrate
                )
                close = eligible[yawrate_error <= 0.02]
                if len(close):
                    eligible = close
                else:
                    eligible = eligible[np.argsort(yawrate_error)[:20]]
                ordered = sorted(
                    eligible.tolist(),
                    key=lambda index: stable_hash(
                        args.seed,
                        speed,
                        variant_index,
                        repeat_index,
                        desired_yawrate,
                        str(seed_data["source_file"][index]),
                        int(seed_data["source_window_index"][index]),
                    ),
                )
                seed_row = ordered[0]
                rows.append(
                    {
                        "episode_index": len(rows),
                        "episode_id": f"episode_{len(rows):03d}",
                        "speed_kph": speed,
                        "variant_index": variant_index,
                        "repeat_index": repeat_index,
                        "road_variation_slot": int(variant["road_variation_slot"]),
                        "seed_row": seed_row,
                        "desired_seed_yawrate_rps": desired_yawrate,
                        "seed_yawrate_rps": float(seed_data["state_six"][seed_row, 5]),
                        "desired_seed_source": desired_source,
                    }
                )
    requested_indices = {
        int(value) for value in args.episode_indices.split(",") if value.strip()
    }
    if requested_indices:
        available_indices = {int(row["episode_index"]) for row in rows}
        missing_indices = requested_indices.difference(available_indices)
        if missing_indices:
            raise ValueError(
                f"episode indices outside the full plan: {sorted(missing_indices)}"
            )
        rows = [
            row for row in rows if int(row["episode_index"]) in requested_indices
        ]
    elif args.max_episodes > 0:
        rows = rows[: args.max_episodes]
    return rows


def collect_episode(
    episode: dict,
    seed_data: np.lib.npyio.NpzFile,
    model: QueryDeploymentModel,
    args: argparse.Namespace,
    root: Path,
) -> dict:
    episode_dir = root / episode["episode_id"]
    if episode_dir.exists():
        raise FileExistsError(f"refusing to replace {episode_dir}")
    episode_dir.mkdir()

    speed_mps = float(episode["speed_kph"]) / 3.6
    variant = effective_variant(
        int(episode["variant_index"]),
        int(episode["speed_kph"]),
        int(episode.get("repeat_index", 0)),
        args.repeats_per_cell,
        args.road_variation_fraction,
    )
    waypoints, road_geometry = make_waypoints(variant, speed_mps)
    road = PeriodicArcLengthRoad(waypoints)
    row = int(episode["seed_row"])
    seed_state = np.asarray(
        seed_data["state_six"][row, (0, 1, 2, 3, 5)], np.float32
    )
    history = np.asarray(seed_data["history"][row], np.float32).copy()
    current_action = np.asarray(seed_data["current_action"][row], np.float32).copy()
    state, start_s = initial_state_on_road(
        road,
        seed_state,
        variant,
        start_fraction=0.071 + 0.113 * int(episode["episode_index"]),
    )

    params = TorchMPPIParams(
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        sampling_mode=args.sampling_mode,
        seed=args.seed + 1009 * int(episode["episode_index"]),
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(model), params, device=args.device
    )
    running = controller.get_init_state(current_action)
    last_snapshot_step = args.burn_in_steps + (
        args.snapshots_per_episode - 1
    ) * args.snapshot_stride
    total_steps = last_snapshot_step + 1
    snapshot_steps = set(
        range(
            args.burn_in_steps,
            last_snapshot_step + 1,
            args.snapshot_stride,
        )
    )

    trace: dict[str, list] = {
        name: []
        for name in (
            "control_step",
            "state",
            "current_action",
            "reference",
            "reference_ego",
            "mean_knots_before",
            "mean_knots_after",
            "executed_action",
            "next_state",
            "weighted_output_cost",
            "best_candidate_cost",
        )
    }
    snapshots: dict[str, list] = {
        name: []
        for name in (
            "control_step",
            "state",
            "current_action",
            "history",
            "reference",
            "reference_ego",
            "mean_knots_before",
            "mean_knots_after",
            "sampling_noise_knots",
            "sampling_mean_knots",
            "raw_sampled_knots",
            "sampled_knots",
            "sampled_action_sequences",
            "predicted_trajectories",
            "cost",
            "weight",
            "optimized_action_sequence",
            "optimized_trajectory",
            "optimized_cost",
        )
    }
    component_snapshots: dict[str, list] = {}
    feature_snapshots: dict[str, list] = {}
    seed_history = history.copy()
    seed_initial_state = state.copy()
    seed_current_action = current_action.copy()

    for step in range(total_steps):
        reference, _projection = road.reference(
            state,
            speed_mps,
            params.dt,
            params.horizon + 1,
        )
        reference = np.asarray(reference, np.float32)
        before = running.mean_knots.detach().cpu().numpy().copy()
        action, running_after, info = controller(
            state,
            current_action,
            history[None],
            reference,
            running,
        )
        optimized_sequence = info["optimized_action_sequence"]
        optimized = controller.evaluate_action_sequences(
            state,
            current_action,
            history[None],
            reference,
            optimized_sequence,
        )
        next_state = optimized["trajectories"][0, 0].detach().cpu().numpy()
        executed_action = action.detach().cpu().numpy()
        optimized_cost = float(optimized["cost"][0].detach().cpu())

        trace["control_step"].append(step)
        trace["state"].append(state.copy())
        trace["current_action"].append(current_action.copy())
        trace["reference"].append(reference)
        trace["reference_ego"].append(reference_ego(reference, state))
        trace["mean_knots_before"].append(before)
        trace["mean_knots_after"].append(
            running_after.mean_knots.detach().cpu().numpy()
        )
        trace["executed_action"].append(executed_action)
        trace["next_state"].append(next_state)
        trace["weighted_output_cost"].append(optimized_cost)
        trace["best_candidate_cost"].append(float(info["best_cost"].cpu()))

        if step in snapshot_steps:
            for name, value in (
                ("control_step", step),
                ("state", state.copy()),
                ("current_action", current_action.copy()),
                ("history", history.copy()),
                ("reference", reference),
                ("reference_ego", reference_ego(reference, state)),
                ("mean_knots_before", before),
                (
                    "mean_knots_after",
                    running_after.mean_knots.detach().cpu().numpy(),
                ),
                (
                    "sampling_noise_knots",
                    info["sampling_noise_knots"].cpu().numpy(),
                ),
                (
                    "sampling_mean_knots",
                    info["sampling_mean_knots"].cpu().numpy(),
                ),
                ("raw_sampled_knots", info["raw_sampled_knots"].cpu().numpy()),
                ("sampled_knots", info["sampled_knots"].cpu().numpy()),
                (
                    "sampled_action_sequences",
                    info["sampled_action_sequences"].cpu().numpy(),
                ),
                (
                    "predicted_trajectories",
                    info["sampled_trajectories"].cpu().numpy(),
                ),
                ("cost", info["cost"].cpu().numpy()),
                ("weight", info["weight"].cpu().numpy()),
                ("optimized_action_sequence", optimized_sequence.cpu().numpy()),
                ("optimized_trajectory", optimized["trajectories"][0].cpu().numpy()),
                ("optimized_cost", optimized_cost),
            ):
                snapshots[name].append(value)
            for name, value in info["cost_components"].items():
                component_snapshots.setdefault(name, []).append(value.cpu().numpy())
            candidate_trajectory = info["sampled_trajectories"].cpu().numpy()
            candidate_actions = info["sampled_action_sequences"].cpu().numpy()
            for name, value in raw_features(
                candidate_trajectory,
                candidate_actions,
                reference,
                current_action,
            ).items():
                feature_snapshots.setdefault(name, []).append(value)

        history = append_history(history, state, next_state, executed_action)
        state = next_state.astype(np.float32)
        current_action = executed_action.astype(np.float32)
        running = running_after

    trace_arrays = {name: np.asarray(value) for name, value in trace.items()}
    snapshot_arrays = {
        name: np.asarray(value) for name, value in snapshots.items()
    }
    for name, value in component_snapshots.items():
        snapshot_arrays[f"cost_component_{name}"] = np.asarray(value)
    for name, value in feature_snapshots.items():
        snapshot_arrays[name] = np.asarray(value)
    np.savez_compressed(
        episode_dir / "road.npz",
        source_waypoints=waypoints.astype(np.float32),
        arc_length_nodes=road.s_nodes.astype(np.float64),
    )
    np.savez_compressed(
        episode_dir / "trace.npz",
        **trace_arrays,
        seed_history=seed_history,
        seed_initial_state=seed_initial_state,
        seed_current_action=seed_current_action,
    )
    np.savez_compressed(episode_dir / "snapshots.npz", **snapshot_arrays)

    yawrate = trace_arrays["state"][:, 4]
    speed = trace_arrays["state"][:, 3]
    reference_position_error = np.linalg.norm(
        trace_arrays["state"][:, :2] - trace_arrays["reference"][:, 0, :2],
        axis=1,
    )
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "episode": episode,
        "road": {
            **road_geometry,
            "total_length_m": road.total_length,
            "initial_frenet_s_m": start_s,
        },
        "seed": {
            "source": str(seed_data["source"][row]),
            "source_file": str(seed_data["source_file"][row]),
            "source_window_index": int(seed_data["source_window_index"][row]),
            "seed_replay_row": row,
        },
        "trace_steps": total_steps,
        "snapshot_count": len(snapshot_arrays["control_step"]),
        "actual_speed_mps": stats(speed),
        "actual_yawrate_rps": stats(yawrate),
        "reference_position_error_m": stats(reference_position_error),
        "weighted_output_cost": stats(trace_arrays["weighted_output_cost"]),
        "artifacts": {
            name: {"path": name, "sha256": sha256(episode_dir / name)}
            for name in ("road.npz", "trace.npz", "snapshots.npz")
        },
    }
    json_dump(episode_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.episode_indices.strip() and args.max_episodes > 0:
        raise ValueError("--episode-indices and --max-episodes are mutually exclusive")
    if args.episodes_per_speed < 1 or args.episodes_per_speed > len(ROAD_VARIANTS):
        raise ValueError(
            f"--episodes-per-speed must be in [1,{len(ROAD_VARIANTS)}]"
        )
    if args.repeats_per_cell < 1:
        raise ValueError("--repeats-per-cell must be positive")
    if not 0.0 <= args.road_variation_fraction <= 0.25:
        raise ValueError("--road-variation-fraction must be in [0,0.25]")
    if args.burn_in_steps < 250:
        raise ValueError("collection requires at least 250 closed-loop burn-in steps")
    if args.snapshots_per_episode < 1 or args.snapshot_stride < 1:
        raise ValueError("snapshot count and stride must be positive")
    if args.sampling_mode == "fixed_hadamard_64" and args.num_samples != 64:
        raise ValueError("fixed_hadamard_64 requires --num-samples=64")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = args.checkpoint.resolve()
    seed_replay = args.seed_replay.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    seed_data = np.load(seed_replay, allow_pickle=False)
    required = {
        "source",
        "source_file",
        "source_window_index",
        "speed_bin_kph",
        "state_six",
        "history",
        "current_action",
    }
    missing = required.difference(seed_data.files)
    if missing:
        raise KeyError(f"seed replay missing {sorted(missing)}")
    plan = choose_seed_rows(seed_data, args)
    collection_manifest = {
        "format_version": 1,
        "dataset_type": "anycar-query-expected-road-closed-loop",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "train-only Query-environment MPPI collection",
        "query_checkpoint": str(checkpoint),
        "query_checkpoint_sha256": sha256(checkpoint),
        "seed_replay": str(seed_replay),
        "seed_replay_sha256": sha256(seed_replay),
        "seed_replay_fields_used": [
            "history",
            "state_six[x,y,yaw,vx,yawrate]",
            "current_action",
            "source provenance",
            "speed bin",
        ],
        "seed_replay_fields_explicitly_not_used": [
            "reference",
            "behavior_future_state",
            "behavior_future_action",
            "mean_knots_before",
            "cost",
        ],
        "causal_contract": (
            "step 0 repeats current_action as the cold-start knot center; each "
            "later step uses the preceding controller call's shifted output knots"
        ),
        "reference_contract": (
            "periodic arc-length desired centerline projection plus constant desired "
            "speed; never the recorded future trajectory"
        ),
        "plant_contract": (
            "the first state of a fresh frozen-Query rollout of the optimized "
            "action sequence is the deterministic next state"
        ),
        "collection": {
            "burn_in_steps": args.burn_in_steps,
            "snapshots_per_episode": args.snapshots_per_episode,
            "snapshot_stride": args.snapshot_stride,
            "repeats_per_cell": args.repeats_per_cell,
            "road_variation_fraction": args.road_variation_fraction,
            "speed_bins_kph": SPEED_BINS_KPH,
            "base_road_variants": ROAD_VARIANTS,
            "selected_variant_indices": selected_variant_indices(args),
            "mppi": asdict(
                TorchMPPIParams(
                    num_samples=args.num_samples,
                    num_iterations=args.num_iterations,
                    sampling_mode=args.sampling_mode,
                    seed=args.seed,
                )
            ),
            "sampling_bank_version": (
                FIXED_HADAMARD_BANK_VERSION
                if args.sampling_mode == "fixed_hadamard_64"
                else None
            ),
        },
        "formal_validation_or_test_consumed": False,
        "episodes": plan,
    }
    json_dump(output / "manifest.json", collection_manifest)

    model = QueryDeploymentModel.from_checkpoint(checkpoint, args.device)
    episode_summaries = []
    for episode in plan:
        print(
            f"collecting {episode['episode_id']} speed={episode['speed_kph']} "
            f"variant={episode['variant_index']}",
            flush=True,
        )
        episode_summaries.append(
            collect_episode(episode, seed_data, model, args, output)
        )
    all_speed = np.concatenate(
        [
            np.load(output / row["episode"]["episode_id"] / "trace.npz")["state"][:, 3]
            for row in episode_summaries
        ]
    )
    all_yawrate = np.concatenate(
        [
            np.load(output / row["episode"]["episode_id"] / "trace.npz")["state"][:, 4]
            for row in episode_summaries
        ]
    )
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "episode_count": len(episode_summaries),
        "snapshot_count": int(
            sum(row["snapshot_count"] for row in episode_summaries)
        ),
        "actual_speed_mps": stats(all_speed),
        "actual_speed_kph": stats(all_speed * 3.6),
        "actual_yawrate_rps": stats(all_yawrate),
        "episode_summaries": episode_summaries,
        "formal_validation_or_test_consumed": False,
    }
    json_dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
