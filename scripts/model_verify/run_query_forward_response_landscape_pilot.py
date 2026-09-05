#!/usr/bin/env python3
"""Run the frozen 20-state pure-Query forward-response landscape pilot.

Five independent initial branches are retained for four rounds.  Every round
uses 32 same-state antithetic Query probes to fit scalar-cost and full weighted
trajectory-residual responses, then evaluates six response proposals with the
real frozen Query rollout.  The old warm branch is shadow-only and can never
enter the four-branch canonical winner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_forward_response_landscape_pilot_config_20260902_v1.json"
)
CONTEXT_FIELDS = (
    "row_index",
    "episode_id",
    "episode_index",
    "row_in_episode",
    "control_step",
    "speed_kph",
    "speed_index",
    "variant_index",
    "road_name",
    "fold_id",
    "source_snapshot_sha256",
    "state",
    "current_action",
    "history",
    "reference",
    "reference_ego",
)


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


def dump_json(path: Path, value: object) -> None:
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


def hadamard_16() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < 16:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def dct_16() -> np.ndarray:
    index = np.arange(16, dtype=np.float64)
    frequency = np.arange(16, dtype=np.float64)[:, None]
    matrix = np.cos(math.pi * (index[None, :] + 0.5) * frequency / 16.0)
    matrix[0] *= math.sqrt(1.0 / 16.0)
    matrix[1:] *= math.sqrt(2.0 / 16.0)
    return matrix * math.sqrt(16.0)


def qr_16(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((16, 16))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return (q * signs[None, :]).T * math.sqrt(16.0)


def basis_bank() -> np.ndarray:
    bases = np.stack((hadamard_16(), dct_16(), qr_16(260902), qr_16(260903)))
    rms = np.sqrt(np.mean(np.square(bases), axis=2))
    if not np.allclose(rms, 1.0, atol=1e-12):
        raise AssertionError("basis RMS contract failed")
    if any(np.linalg.matrix_rank(value) != 16 for value in bases):
        raise AssertionError("basis rank contract failed")
    return bases.astype(np.float32)


def interpolate_knots(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    return (
        F.interpolate(
            tensor.transpose(1, 2), size=50, mode="linear", align_corners=True
        )
        .transpose(1, 2)
        .numpy()
    )


def weighted_residual(
    trajectories: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    current_action: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    target = reference[1:]
    yaw_delta = trajectories[..., 2] - target[None, :, 2]
    yaw_delta = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    previous = np.concatenate(
        (
            np.broadcast_to(current_action, (len(actions), 1, 2)),
            actions[:, :-1],
        ),
        axis=1,
    )
    rate = actions - previous
    fields = (
        math.sqrt(weights["position"])
        * (trajectories[..., 0:2] - target[None, :, 0:2]),
        math.sqrt(weights["yaw"]) * yaw_delta[..., None],
        math.sqrt(weights["vx"])
        * (trajectories[..., 3] - target[None, :, 3])[..., None],
        math.sqrt(weights["acceleration_rate"]) * rate[..., 0:1],
        math.sqrt(weights["steering_rate"]) * rate[..., 1:2],
    )
    return np.concatenate(fields, axis=2).reshape(len(actions), -1).astype(np.float32)


def evaluate(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    knots: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    actions = interpolate_knots(knots)
    result = controller.evaluate_action_sequences(
        data["state"][row],
        data["current_action"][row],
        data["history"][row : row + 1],
        data["reference"][row],
        actions,
    )
    cost = result["cost"].cpu().numpy().astype(np.float32)
    trajectories = result["trajectories"].cpu().numpy().astype(np.float32)
    residual = weighted_residual(
        trajectories,
        actions,
        data["reference"][row],
        data["current_action"][row],
        weights,
    )
    error = float(np.max(np.abs(np.sum(residual.astype(np.float64) ** 2, axis=1) - cost)))
    if error > 5e-3:
        raise AssertionError(f"residual/cost reconstruction error {error}")
    return cost, residual


def stable_solve(matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, right, rcond=None)[0]


def normalize_rms(vector: np.ndarray) -> np.ndarray:
    rho = float(np.sqrt(np.mean(np.square(vector))))
    return vector / rho if rho > 1e-10 else np.zeros_like(vector)


def fit_response(
    center: np.ndarray,
    center_cost: float,
    center_residual: np.ndarray,
    probe_knots: np.ndarray,
    probe_cost: np.ndarray,
    probe_residual: np.ndarray,
    sigma: np.ndarray,
    ridge: float,
    damping: float,
    maximum_step: float,
) -> dict[str, np.ndarray | float]:
    x_probe = ((probe_knots - center[None]) / sigma[None, None, :]).reshape(-1, 16)
    x = np.concatenate((np.zeros((1, 16)), x_probe), axis=0).astype(np.float64)
    delta_cost = np.concatenate(([0.0], probe_cost.astype(np.float64) - center_cost))
    delta_residual = np.concatenate(
        (
            np.zeros((1, center_residual.size), dtype=np.float64),
            probe_residual.astype(np.float64) - center_residual[None].astype(np.float64),
        ),
        axis=0,
    )
    scale = max(float(np.median(np.abs(delta_cost))), 0.25)
    weight = 1.0 / (1.0 + np.square(np.abs(delta_cost) / scale))
    weight /= float(np.mean(weight))
    lhs = x.T @ (weight[:, None] * x) + ridge * np.eye(16)
    gradient = stable_solve(lhs, x.T @ (weight * delta_cost))
    response = stable_solve(lhs, x.T @ (weight[:, None] * delta_residual))

    gn_lhs = response @ response.T + damping * np.eye(16)
    gn_step = -stable_solve(gn_lhs, response @ center_residual.astype(np.float64))
    gn_rho_uncapped = float(np.sqrt(np.mean(np.square(gn_step))))
    if gn_rho_uncapped > maximum_step:
        gn_step *= maximum_step / gn_rho_uncapped
    cost_direction = normalize_rms(-gradient)
    trajectory_direction = normalize_rms(gn_step)
    cosine = float(np.mean(cost_direction * trajectory_direction))
    if cosine < 0.0:
        trajectory_direction = -trajectory_direction
    blend_direction = normalize_rms(cost_direction + trajectory_direction)

    predicted_cost = x @ gradient
    cost_denominator = max(float(np.sum(weight * np.square(delta_cost))), 1e-8)
    cost_fit_error = math.sqrt(
        float(np.sum(weight * np.square(predicted_cost - delta_cost)))
        / cost_denominator
    )
    predicted_residual = x @ response
    residual_denominator = max(
        float(np.sum(weight[:, None] * np.square(delta_residual))), 1e-8
    )
    residual_fit_error = math.sqrt(
        float(
            np.sum(weight[:, None] * np.square(predicted_residual - delta_residual))
        )
        / residual_denominator
    )
    return {
        "x": x.astype(np.float32),
        "weight": weight.astype(np.float32),
        "gradient": gradient.astype(np.float32),
        "response": response.astype(np.float32),
        "gn_step": gn_step.astype(np.float32),
        "cost_direction": cost_direction.astype(np.float32),
        "trajectory_direction": trajectory_direction.astype(np.float32),
        "blend_direction": blend_direction.astype(np.float32),
        "cost_fit_error": cost_fit_error,
        "residual_fit_error": residual_fit_error,
        "direction_cosine_before_alignment": cosine,
        "gn_step_rho_uncapped": gn_rho_uncapped,
    }


def select_rows(
    archive: np.lib.npyio.NpzFile, config: dict
) -> np.ndarray:
    selected = []
    overrides = {(0, 1): 3, (0, 3): 1}
    for speed_index in range(5):
        for variant in range(4):
            fold = overrides.get((speed_index, variant), (speed_index + variant) % 5)
            candidates = np.flatnonzero(
                (archive["speed_index"] == speed_index)
                & (archive["variant_index"] == variant)
                & (archive["fold_id"] == fold)
            )
            if len(candidates) != 1:
                raise AssertionError(
                    f"expected one source row for cell {(speed_index, variant, fold)}, got {len(candidates)}"
                )
            selected.append(int(candidates[0]))
    rows = np.asarray(selected, np.int64)
    expected = int(config["state_selection"]["expected_state_count"])
    if len(rows) != expected:
        raise AssertionError("state-count contract failed")
    return rows


def find_collection(source_manifest: dict) -> tuple[Path, Path]:
    sidecar = Path(source_manifest["source_sidecar"])
    sidecar_manifest = json.loads((sidecar / "manifest.json").read_text())
    parent_t0 = Path(sidecar_manifest["parent_t0"])
    t0_manifest = json.loads((parent_t0 / "manifest.json").read_text())
    return Path(t0_manifest["source_collection"]), parent_t0


def basin_count(centers: np.ndarray, costs: np.ndarray, sigma: np.ndarray) -> tuple[int, float]:
    eligible = np.flatnonzero(costs <= float(np.min(costs)) * 1.10 + 1e-8)
    retained: list[int] = []
    for index in eligible[np.argsort(costs[eligible])]:
        if all(
            np.sqrt(np.mean(np.square((centers[index] - centers[other]) / sigma)))
            >= 0.5
            for other in retained
        ):
            retained.append(int(index))
    separation = 0.0
    if len(retained) > 1:
        separation = min(
            float(np.sqrt(np.mean(np.square((centers[a] - centers[b]) / sigma))))
            for pos, a in enumerate(retained)
            for b in retained[pos + 1 :]
        )
    return len(retained), separation


def grouped_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    final = data["canonical_final_cost"][mask].astype(np.float64)
    warm = data["old_warm_cost"][mask].astype(np.float64)
    prior = data["prior_canonical_cost"][mask].astype(np.float64)
    return {
        "rows": int(mask.sum()),
        "canonical_final_cost": stats(final),
        "old_warm_cost": stats(warm),
        "prior_canonical_cost": stats(prior),
        "gain_vs_warm": stats(warm - final),
        "gain_vs_prior_canonical": stats(prior - final),
        "warm_regression_fraction": float(np.mean(final > warm + 1e-5)),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    source = Path(config["source_audit"]).resolve()
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    if config.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("DBM fields or labels are forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_manifest_path = source / "manifest.json"
    source_validation_path = source / "validation.json"
    source_audit_path = source / "audit.npz"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_validation = json.loads(source_validation_path.read_text())
    if source_validation["qualification"] != "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS":
        raise AssertionError("source canonical audit did not independently pass")
    if source_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source consumed formal validation/test")
    if sha256(source_audit_path) != source_manifest["audit_sha256"]:
        raise AssertionError("source audit hash mismatch")

    with np.load(source_audit_path, allow_pickle=False) as archive:
        source_rows = select_rows(archive, config)
        data = {name: np.asarray(archive[name][source_rows]) for name in CONTEXT_FIELDS}
        data.update(
            {
                "source_audit_row": source_rows,
                "old_warm_knots": np.asarray(archive["old_warm_knots"][source_rows]),
                "old_warm_cost": np.asarray(archive["old_warm_cost"][source_rows]),
                "old_t0_cost": np.asarray(archive["old_t0_cost"][source_rows]),
                "old_fullrank_cost": np.asarray(archive["old_fullrank_cost"][source_rows]),
                "prior_canonical_knots": np.asarray(
                    archive["canonical_teacher_knots"][source_rows]
                ),
                "prior_canonical_cost": np.asarray(
                    archive["canonical_teacher_cost"][source_rows]
                ),
            }
        )
    if len(np.unique(data["episode_id"])) != len(source_rows):
        raise AssertionError("selected states are not episode independent")
    if [int(np.sum(data["fold_id"] == fold)) for fold in range(5)] != [4] * 5:
        raise AssertionError("selected fold balance failed")
    if "episode_028" not in set(data["episode_id"].tolist()):
        raise AssertionError("episode_028 tail was not retained")

    collection, parent_t0 = find_collection(source_manifest)
    collection_manifest_path = collection / "manifest.json"
    collection_manifest = json.loads(collection_manifest_path.read_text())
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    query_checkpoint = Path(source_manifest["query_checkpoint"])
    query_model = QueryDeploymentModel.from_checkpoint(query_checkpoint, device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), params, device=str(device)
    )

    branch_names = np.asarray([value["name"] for value in config["branches"]])
    branch_roles = np.asarray([value["role"] for value in config["branches"]])
    canonical_branch = branch_roles == "canonical"
    branch_count = len(branch_names)
    round_count = int(config["round_count"])
    row_count = len(source_rows)
    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    bases = basis_bank()
    radii = np.asarray(config["probe_contract"]["radius_sigma_by_round"], np.float32)
    maximum_steps = np.asarray(
        config["response_contract"]["maximum_step_sigma_by_round"], np.float32
    )
    line_factors = np.asarray(config["response_contract"]["line_factors"], np.float32)
    ridge = float(config["response_contract"]["fit_ridge"])
    damping = float(config["response_contract"]["gauss_newton_damping"])
    probe_count = 32
    proposal_count = 6
    residual_dim = 300

    centers = np.empty((row_count, branch_count, round_count + 1, 8, 2), np.float32)
    center_cost = np.empty((row_count, branch_count, round_count + 1), np.float32)
    center_residual = np.empty(
        (row_count, branch_count, round_count + 1, residual_dim), np.float32
    )
    probe_raw = np.empty((row_count, branch_count, round_count, probe_count, 8, 2), np.float32)
    probe_knots = np.empty_like(probe_raw)
    probe_cost = np.empty((row_count, branch_count, round_count, probe_count), np.float32)
    probe_residual = np.empty(
        (row_count, branch_count, round_count, probe_count, residual_dim), np.float32
    )
    probe_clipped = np.empty_like(probe_raw, dtype=bool)
    fit_x = np.empty((row_count, branch_count, round_count, 33, 16), np.float32)
    fit_weight = np.empty((row_count, branch_count, round_count, 33), np.float32)
    cost_gradient = np.empty((row_count, branch_count, round_count, 16), np.float32)
    trajectory_response = np.empty(
        (row_count, branch_count, round_count, 16, residual_dim), np.float32
    )
    gn_step = np.empty_like(cost_gradient)
    response_directions = np.empty((row_count, branch_count, round_count, 3, 16), np.float32)
    cost_fit_error = np.empty((row_count, branch_count, round_count), np.float32)
    residual_fit_error = np.empty_like(cost_fit_error)
    direction_cosine = np.empty_like(cost_fit_error)
    gn_rho_uncapped = np.empty_like(cost_fit_error)
    proposal_delta_sigma = np.empty(
        (row_count, branch_count, round_count, proposal_count, 16), np.float32
    )
    proposal_raw = np.empty(
        (row_count, branch_count, round_count, proposal_count, 8, 2), np.float32
    )
    proposal_knots = np.empty_like(proposal_raw)
    proposal_cost = np.empty(
        (row_count, branch_count, round_count, proposal_count), np.float32
    )
    proposal_residual = np.empty(
        (row_count, branch_count, round_count, proposal_count, residual_dim), np.float32
    )
    proposal_clipped = np.empty_like(proposal_raw, dtype=bool)
    proposal_predicted_scalar_cost = np.empty_like(proposal_cost)
    proposal_predicted_trajectory_cost = np.empty_like(proposal_cost)
    selected_source = np.empty((row_count, branch_count, round_count), np.int8)
    selected_local_index = np.empty((row_count, branch_count, round_count), np.int16)
    probe_only_best_cost = np.empty_like(cost_fit_error)
    response_incremental_gain = np.empty_like(cost_fit_error)

    for row in range(row_count):
        current = data["current_action"][row]
        initial = np.stack(
            (
                np.broadcast_to(current, (8, 2)),
                np.linspace(current, np.zeros(2, np.float32), 8),
                np.zeros((8, 2), np.float32),
                data["prior_canonical_knots"][row],
                data["old_warm_knots"][row],
            )
        ).astype(np.float32)
        initial = np.clip(initial, low, high)
        initial_cost, initial_residual = evaluate(controller, data, row, initial, weights)
        centers[row, :, 0] = initial
        center_cost[row, :, 0] = initial_cost
        center_residual[row, :, 0] = initial_residual

        for round_index in range(round_count):
            basis = bases[round_index].reshape(16, 8, 2)
            radius = float(radii[round_index])
            maximum_step = float(maximum_steps[round_index])
            for branch in range(branch_count):
                incumbent = centers[row, branch, round_index]
                incumbent_cost = float(center_cost[row, branch, round_index])
                incumbent_residual = center_residual[row, branch, round_index]
                raw = np.stack(
                    [
                        incumbent + sign * radius * direction * sigma[None]
                        for direction in basis
                        for sign in (1.0, -1.0)
                    ]
                ).astype(np.float32)
                local = np.clip(raw, low, high)
                local_cost, local_residual = evaluate(
                    controller, data, row, local, weights
                )
                probe_raw[row, branch, round_index] = raw
                probe_knots[row, branch, round_index] = local
                probe_cost[row, branch, round_index] = local_cost
                probe_residual[row, branch, round_index] = local_residual
                probe_clipped[row, branch, round_index] = np.abs(raw - local) > 1e-7

                fitted = fit_response(
                    incumbent,
                    incumbent_cost,
                    incumbent_residual,
                    local,
                    local_cost,
                    local_residual,
                    sigma,
                    ridge,
                    damping,
                    maximum_step,
                )
                fit_x[row, branch, round_index] = fitted["x"]
                fit_weight[row, branch, round_index] = fitted["weight"]
                cost_gradient[row, branch, round_index] = fitted["gradient"]
                trajectory_response[row, branch, round_index] = fitted["response"]
                gn_step[row, branch, round_index] = fitted["gn_step"]
                response_directions[row, branch, round_index] = np.stack(
                    (
                        fitted["cost_direction"],
                        fitted["trajectory_direction"],
                        fitted["blend_direction"],
                    )
                )
                cost_fit_error[row, branch, round_index] = fitted["cost_fit_error"]
                residual_fit_error[row, branch, round_index] = fitted["residual_fit_error"]
                direction_cosine[row, branch, round_index] = fitted[
                    "direction_cosine_before_alignment"
                ]
                gn_rho_uncapped[row, branch, round_index] = fitted["gn_step_rho_uncapped"]

                steps = []
                for direction_index in range(3):
                    base_step = (
                        fitted["gn_step"]
                        if direction_index == 1
                        else maximum_step
                        * response_directions[row, branch, round_index, direction_index]
                    )
                    for factor in line_factors:
                        steps.append(float(factor) * base_step)
                steps = np.stack(steps).astype(np.float32)
                raw_proposals = incumbent[None] + steps.reshape(-1, 8, 2) * sigma
                local_proposals = np.clip(raw_proposals, low, high).astype(np.float32)
                local_proposal_cost, local_proposal_residual = evaluate(
                    controller, data, row, local_proposals, weights
                )
                proposal_delta_sigma[row, branch, round_index] = steps
                proposal_raw[row, branch, round_index] = raw_proposals
                proposal_knots[row, branch, round_index] = local_proposals
                proposal_cost[row, branch, round_index] = local_proposal_cost
                proposal_residual[row, branch, round_index] = local_proposal_residual
                proposal_clipped[row, branch, round_index] = (
                    np.abs(raw_proposals - local_proposals) > 1e-7
                )
                proposal_predicted_scalar_cost[row, branch, round_index] = (
                    incumbent_cost + steps @ fitted["gradient"]
                )
                predicted_residual = (
                    incumbent_residual[None]
                    + steps.astype(np.float64) @ fitted["response"].astype(np.float64)
                )
                proposal_predicted_trajectory_cost[row, branch, round_index] = np.sum(
                    np.square(predicted_residual), axis=1
                )

                probe_best = min(incumbent_cost, float(np.min(local_cost)))
                proposal_best = float(np.min(local_proposal_cost))
                probe_only_best_cost[row, branch, round_index] = probe_best
                response_incremental_gain[row, branch, round_index] = max(
                    0.0, probe_best - proposal_best
                )
                combined_cost = np.concatenate(
                    ([incumbent_cost], local_cost, local_proposal_cost)
                )
                winner = int(np.argmin(combined_cost))
                if winner == 0:
                    next_center = incumbent
                    next_residual = incumbent_residual
                    source_code, local_index = 0, 0
                elif winner <= probe_count:
                    local_index = winner - 1
                    next_center = local[local_index]
                    next_residual = local_residual[local_index]
                    source_code = 1
                else:
                    local_index = winner - 1 - probe_count
                    next_center = local_proposals[local_index]
                    next_residual = local_proposal_residual[local_index]
                    source_code = 2
                centers[row, branch, round_index + 1] = next_center
                center_cost[row, branch, round_index + 1] = combined_cost[winner]
                center_residual[row, branch, round_index + 1] = next_residual
                selected_source[row, branch, round_index] = source_code
                selected_local_index[row, branch, round_index] = local_index
        print(
            f"forward-response state {row + 1}/{row_count} "
            f"speed={int(data['speed_kph'][row])} episode={data['episode_id'][row]}",
            flush=True,
        )

    canonical_final_branch = np.argmin(center_cost[:, canonical_branch, -1], axis=1)
    canonical_indices = np.flatnonzero(canonical_branch)
    canonical_global_branch = canonical_indices[canonical_final_branch]
    rows = np.arange(row_count)
    canonical_final_cost = center_cost[rows, canonical_global_branch, -1]
    canonical_final_knots = centers[rows, canonical_global_branch, -1]
    basin_counts = np.empty(row_count, np.int64)
    basin_min_separation = np.empty(row_count, np.float32)
    for row in range(row_count):
        basin_counts[row], basin_min_separation[row] = basin_count(
            centers[row, canonical_branch, -1],
            center_cost[row, canonical_branch, -1],
            sigma,
        )
    data.update(
        {
            "branch_name": branch_names,
            "branch_role": branch_roles,
            "round_basis": bases,
            "round_radius_sigma": radii,
            "round_maximum_step_sigma": maximum_steps,
            "line_factor": line_factors,
            "center_knots": centers,
            "center_cost": center_cost,
            "center_residual": center_residual,
            "probe_raw_knots": probe_raw,
            "probe_knots": probe_knots,
            "probe_cost": probe_cost,
            "probe_residual": probe_residual,
            "probe_clipped_mask": probe_clipped,
            "fit_x": fit_x,
            "fit_weight": fit_weight,
            "cost_gradient": cost_gradient,
            "trajectory_response": trajectory_response,
            "gauss_newton_step_sigma": gn_step,
            "response_direction": response_directions,
            "cost_relative_fit_error": cost_fit_error,
            "trajectory_relative_fit_error": residual_fit_error,
            "cost_trajectory_direction_cosine_before_alignment": direction_cosine,
            "gauss_newton_step_rho_uncapped": gn_rho_uncapped,
            "proposal_delta_sigma": proposal_delta_sigma,
            "proposal_raw_knots": proposal_raw,
            "proposal_knots": proposal_knots,
            "proposal_cost": proposal_cost,
            "proposal_residual": proposal_residual,
            "proposal_clipped_mask": proposal_clipped,
            "proposal_predicted_scalar_cost": proposal_predicted_scalar_cost,
            "proposal_predicted_trajectory_cost": proposal_predicted_trajectory_cost,
            "selected_source": selected_source,
            "selected_local_index": selected_local_index,
            "probe_only_best_cost": probe_only_best_cost,
            "response_incremental_gain": response_incremental_gain,
            "canonical_final_branch": canonical_global_branch,
            "canonical_final_knots": canonical_final_knots,
            "canonical_final_cost": canonical_final_cost,
            "canonical_basin_count_within_10pct_separation_ge_0p5": basin_counts,
            "canonical_basin_min_separation_sigma_rms": basin_min_separation,
        }
    )

    overall = grouped_metrics(data, np.ones(row_count, dtype=bool))
    first_two_mask = np.zeros_like(response_incremental_gain, dtype=bool)
    first_two_mask[:, canonical_branch, :2] = True
    trajectory_proposal_cost = proposal_cost[..., 2:4]
    trajectory_improves = np.min(trajectory_proposal_cost, axis=-1) < probe_only_best_cost - 1e-6
    canonical_context_mask = np.broadcast_to(
        canonical_branch[None, :, None], trajectory_improves.shape
    )
    first_two_response_fraction = float(
        np.mean(response_incremental_gain[first_two_mask] > 1e-6)
    )
    trajectory_improvement_fraction = float(
        np.mean(trajectory_improves[canonical_context_mask])
    )
    aggregate_prior = float(np.sum(data["prior_canonical_cost"]))
    aggregate_gain_vs_prior_fraction = float(
        np.sum(data["prior_canonical_cost"] - canonical_final_cost)
        / aggregate_prior
    )
    gates_config = config["pre_registered_expand_to_100_state_gates"]
    speed_gains = {
        str(speed): float(
            np.mean(
                data["prior_canonical_cost"][data["speed_kph"] == speed]
                - canonical_final_cost[data["speed_kph"] == speed]
            )
        )
        for speed in sorted(np.unique(data["speed_kph"]))
    }
    gates = {
        "canonical_final_mean_cost_no_greater_than_old_warm": bool(
            np.mean(canonical_final_cost) <= np.mean(data["old_warm_cost"]) + 1e-8
        ),
        "canonical_final_warm_regression_fraction_le_0p25": bool(
            overall["warm_regression_fraction"]
            <= float(
                gates_config["canonical_final_warm_regression_fraction_maximum"]
            )
        ),
        "canonical_final_aggregate_gain_vs_prior_canonical_ge_0p10": bool(
            aggregate_gain_vs_prior_fraction
            >= float(
                gates_config[
                    "canonical_final_aggregate_gain_vs_prior_canonical_minimum_fraction"
                ]
            )
        ),
        "first_two_round_response_proposal_improvement_fraction_ge_0p50": bool(
            first_two_response_fraction
            >= float(
                gates_config[
                    "first_two_round_response_proposal_improvement_fraction_minimum"
                ]
            )
        ),
        "trajectory_response_realized_improvement_positive_fraction_ge_0p50": bool(
            trajectory_improvement_fraction
            >= float(
                gates_config[
                    "trajectory_response_realized_improvement_positive_fraction_minimum"
                ]
            )
        ),
        "all_speed_groups_nonnegative_mean_gain_vs_prior_canonical": bool(
            all(value >= -1e-8 for value in speed_gains.values())
        ),
    }
    cost_gates = [
        gates["canonical_final_mean_cost_no_greater_than_old_warm"],
        gates["canonical_final_warm_regression_fraction_le_0p25"],
        gates["canonical_final_aggregate_gain_vs_prior_canonical_ge_0p10"],
        gates["all_speed_groups_nonnegative_mean_gain_vs_prior_canonical"],
    ]
    response_gates = [
        gates["first_two_round_response_proposal_improvement_fraction_ge_0p50"],
        gates["trajectory_response_realized_improvement_positive_fraction_ge_0p50"],
    ]
    if all(cost_gates) and all(response_gates):
        performance_decision = "EXPAND_TO_REMAINING_80_PENDING_INDEPENDENT_REPLAY"
    elif all(response_gates):
        performance_decision = "RESPONSE_WORKS_COST_FAIL_ADJUST_GEOMETRY_ON_SAME_20"
    else:
        performance_decision = "RESPONSE_CONSTRUCTION_FAIL_STOP_NO_LARGE_ONESHOT_BANK"

    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "performance_decision": performance_decision,
        "row_count": row_count,
        "branch_count": branch_count,
        "canonical_branch_count": int(np.sum(canonical_branch)),
        "round_count": round_count,
        "rollouts_per_state": 5 + round_count * branch_count * (probe_count + proposal_count),
        "total_new_query_rollouts": row_count
        * (5 + round_count * branch_count * (probe_count + proposal_count)),
        "overall": overall,
        "center_cost_by_round": {
            str(round_index): stats(
                np.min(center_cost[:, canonical_branch, round_index], axis=1)
            )
            for round_index in range(round_count + 1)
        },
        "branch_final_cost": {
            str(branch_names[branch]): stats(center_cost[:, branch, -1])
            for branch in range(branch_count)
        },
        "response": {
            "first_two_round_proposal_improvement_fraction": first_two_response_fraction,
            "trajectory_proposal_improvement_fraction": trajectory_improvement_fraction,
            "incremental_gain": stats(
                response_incremental_gain[:, canonical_branch]
            ),
            "cost_relative_fit_error": stats(cost_fit_error[:, canonical_branch]),
            "trajectory_relative_fit_error": stats(
                residual_fit_error[:, canonical_branch]
            ),
        },
        "aggregate_gain_vs_prior_canonical_fraction": aggregate_gain_vs_prior_fraction,
        "speed_mean_gain_vs_prior_canonical": speed_gains,
        "canonical_basin_count": stats(basin_counts),
        "pre_registered_performance_gates": gates,
        "by_speed_kph": {
            str(speed): grouped_metrics(data, data["speed_kph"] == speed)
            for speed in sorted(np.unique(data["speed_kph"]))
        },
        "by_variant_index": {
            str(variant): grouped_metrics(data, data["variant_index"] == variant)
            for variant in sorted(np.unique(data["variant_index"]))
        },
        "by_fold": {
            str(fold): grouped_metrics(data, data["fold_id"] == fold)
            for fold in range(5)
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    landscape_path = output / "landscape.npz"
    np.savez_compressed(landscape_path, **data)
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    rows_path = output / "rows.csv"
    with rows_path.open("w", newline="") as stream:
        fields = [
            "source_audit_row",
            "episode_id",
            "speed_kph",
            "variant_index",
            "fold_id",
            "prior_canonical_cost",
            "old_warm_cost",
            "old_t0_cost",
            "old_fullrank_cost",
            "canonical_final_branch",
            "canonical_final_cost",
            "canonical_basin_count_within_10pct_separation_ge_0p5",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in range(row_count):
            writer.writerow(
                {name: np.asarray(data[name][row]).item() for name in fields}
            )
    manifest = {
        "schema_version": "query-forward-response-landscape-pilot-v1",
        "dataset_type": "train-only-pure-query-forward-response-landscape-pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "performance_decision": performance_decision,
        "config_sha256": sha256(config_path),
        "source_audit": str(source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_validation_sha256": sha256(source_validation_path),
        "source_audit_sha256": sha256(source_audit_path),
        "source_collection": str(collection),
        "source_collection_manifest_sha256": sha256(collection_manifest_path),
        "parent_t0": str(parent_t0),
        "query_checkpoint": str(query_checkpoint),
        "query_checkpoint_sha256": sha256(query_checkpoint),
        "landscape_sha256": sha256(landscape_path),
        "summary_sha256": sha256(summary_path),
        "rows_sha256": sha256(rows_path),
        "row_count": row_count,
        "rollouts_per_state": int(summary["rollouts_per_state"]),
        "total_new_query_rollouts": int(summary["total_new_query_rollouts"]),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "response_attestation": (
            "All response fits use only same-state frozen-Query forward probes; "
            "all proposals are rescored by frozen Query and no analytic gradient is used."
        ),
        "shadow_attestation": (
            "old_warm_shadow is retained as an independent branch and comparator only; "
            "it cannot enter canonical branch selection."
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "limitations": [
            "This is a 20-state train-only landscape mechanism pilot, not a teacher dataset.",
            "Query-relative J50 is a model-space objective and is not physical validation.",
            "Independent numerical reconstruction is still required before routing.",
            "No Actor or Critic is trained and formal validation/test remain sealed.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "performance_decision": performance_decision,
                "gates": gates,
                "overall": overall,
                "response": summary["response"],
            },
            indent=2,
        )
    )
    print(f"output: {output}")


if __name__ == "__main__":
    main()
