#!/usr/bin/env python3
"""Train-only strong multi-start center-search oracle for high-speed DBM states."""

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
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from run_mppi_proximal_search_phase1a import ring_candidates, seed_bank_directions


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1"
)
DEFAULT_PRETRAIN = Path(
    "outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2"
)
DEFAULT_OAC = Path(
    "outputs/mppi_proposal/highspeed_actor_visited_oac_expansion_e4_20260830_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32)
LOW = np.asarray((-1.0, -1.0), np.float32)
HIGH = np.asarray((1.0, 1.0), np.float32)
RADII = (1.0, 0.70, 0.50, 0.35, 0.20, 0.10)
START_NAMES = ("warm", "proximal_teacher", "oac_seed0", "oac_seed1", "oac_seed2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--pretrain-dir", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--oac-dir", type=Path, default=DEFAULT_OAC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-chunk", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def load_actor_oof_centers(
    oac_summary: dict, pretrain_summary: dict, count: int,
) -> np.ndarray:
    source_indices = np.asarray(pretrain_summary["contract"]["source_indices"], np.int64)
    if len(source_indices) != count or not np.array_equal(source_indices, np.arange(count)):
        raise AssertionError("strong-search source expects the complete 600-row replay")
    centers = np.full((3, count, 8, 2), np.nan, np.float32)
    for record in oac_summary["records"]:
        seed = int(record["seed"])
        if seed not in (0, 1, 2):
            continue
        checkpoint = Path(record["checkpoint"])
        replay = Path(record["replay"])
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise AssertionError("OAC checkpoint hash mismatch")
        if sha256(replay) != record["replay_sha256"]:
            raise AssertionError("OAC replay hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu")
        rows = np.asarray(payload["oof_indices"], np.int64)
        with np.load(replay, allow_pickle=False) as loaded:
            centers[seed, rows] = loaded["selected_oof_center"]
    if not np.isfinite(centers).all():
        raise AssertionError("OOF Actor centers do not cover all rows and seeds")
    return centers


def select_independent_rows(source: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for speed in sorted(np.unique(source["speed_kph"]).tolist()):
        for scenario in sorted(np.unique(source["scenario_class"]).astype(str).tolist()):
            mask = (
                np.isclose(source["speed_kph"], speed)
                & (source["scenario_class"].astype(str) == scenario)
                & (source["control_step"] == 0)
            )
            local = np.flatnonzero(mask)
            if len(local) != 4 or len(np.unique(source["episode_id"][local])) != 4:
                raise AssertionError(f"expected four independent states for {(speed, scenario)}")
            rows.extend(local.tolist())
    rows = np.asarray(rows, np.int64)
    if len(rows) != 120:
        raise AssertionError("expected 120 strong-search states")
    return rows


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    replay_path = (args.replay_dir / "replay.npz").resolve()
    replay_summary_path = (args.replay_dir / "summary.json").resolve()
    teacher_path = (args.teacher_dir / "labels.npz").resolve()
    teacher_summary_path = (args.teacher_dir / "summary.json").resolve()
    pretrain_summary_path = (args.pretrain_dir / "summary.json").resolve()
    pretrain_validator_path = (args.pretrain_dir / "validator_report.json").resolve()
    oac_summary_path = (args.oac_dir / "summary.json").resolve()
    oac_validator_path = (args.oac_dir / "validator_report.json").resolve()
    replay_summary = json.loads(replay_summary_path.read_text())
    teacher_summary = json.loads(teacher_summary_path.read_text())
    pretrain_summary = json.loads(pretrain_summary_path.read_text())
    pretrain_validator = json.loads(pretrain_validator_path.read_text())
    oac_summary = json.loads(oac_summary_path.read_text())
    oac_validator = json.loads(oac_validator_path.read_text())
    if replay_summary["formal_validation_or_test_created"]:
        raise AssertionError("source replay is not train-only")
    if teacher_summary["protocol"]["formal_validation_or_test_created"]:
        raise AssertionError("source teacher is not train-only")
    if pretrain_validator["qualification"] != "HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("pretraining did not pass independent replay")
    if oac_validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("OAC did not pass independent replay")
    if sha256(replay_path) != replay_summary["archive_sha256"]:
        raise AssertionError("replay hash mismatch")
    if sha256(teacher_path) != teacher_summary["labels_sha256"]:
        raise AssertionError("teacher hash mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(teacher_path, allow_pickle=False) as loaded:
        teacher = {name: np.asarray(loaded[name]) for name in loaded.files}
    count = len(source["state_six"])
    if count != 600 or len(teacher["teacher_knots"]) != count:
        raise AssertionError("unexpected expansion source size")
    actor_centers = load_actor_oof_centers(oac_summary, pretrain_summary, count)
    selected_rows = select_independent_rows(source)

    directions = (
        hadamard_directions().astype(np.float32), seed_bank_directions(2),
        hadamard_directions().astype(np.float32), seed_bank_directions(3),
        hadamard_directions().astype(np.float32), seed_bank_directions(2),
    )
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    start_centers_all, start_costs_all = [], []
    terminal_centers_all, terminal_costs_all = [], []
    path_centers_all, path_costs_all = [], []
    oracle_centers, oracle_costs, evaluation_counts = [], [], []
    evaluated_centers_rows, evaluated_cost_rows = [], []

    for ordinal, row in enumerate(selected_rows):
        state = torch.as_tensor(source["state_six"][row : row + 1], device=device)
        current_action = torch.as_tensor(source["current_action"][row : row + 1], device=device)
        reference = torch.as_tensor(source["reference"][row : row + 1, 1:], device=device)
        cache: dict[bytes, float] = {}
        cache_centers: dict[bytes, np.ndarray] = {}

        def evaluate(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            clipped = np.clip(candidates, LOW, HIGH).astype(np.float32)
            keys = [np.round(value, 7).tobytes() for value in clipped]
            pending_keys, pending = [], []
            for key, value in zip(keys, clipped):
                if key not in cache and key not in pending_keys:
                    pending_keys.append(key); pending.append(value)
            if pending:
                array = np.asarray(pending, np.float32)
                values = []
                with torch.no_grad():
                    for start in range(0, len(array), args.eval_chunk):
                        knots = torch.as_tensor(array[start : start + args.eval_chunk][None], device=device)
                        actions = interpolate_knots(knots, params.horizon)
                        values.append(batched_cost(
                            backend, weights, actions, state, current_action, reference
                        )[0].cpu().numpy())
                for key, center, cost in zip(pending_keys, pending, np.concatenate(values)):
                    cache[key] = float(cost); cache_centers[key] = center.copy()
            return clipped, np.asarray([cache[key] for key in keys], np.float32)

        starts = np.stack((
            source["mean_knots_before"][row], teacher["teacher_knots"][row],
            actor_centers[0, row], actor_centers[1, row], actor_centers[2, row],
        )).astype(np.float32)
        starts, start_costs = evaluate(starts)
        terminal_centers, terminal_costs = [], []
        state_path_centers, state_path_costs = [], []
        for start_center, start_cost in zip(starts, start_costs):
            incumbent = start_center.copy(); incumbent_cost = float(start_cost)
            path_centers, path_costs = [incumbent.copy()], [incumbent_cost]
            for radius, basis in zip(RADII, directions):
                candidates = ring_candidates(incumbent, SIGMA, [radius], basis)
                candidates, costs = evaluate(candidates)
                best = int(np.argmin(costs))
                if float(costs[best]) < incumbent_cost:
                    incumbent = candidates[best].copy(); incumbent_cost = float(costs[best])
                path_centers.append(incumbent.copy()); path_costs.append(incumbent_cost)
            terminal_centers.append(incumbent); terminal_costs.append(incumbent_cost)
            state_path_centers.append(path_centers); state_path_costs.append(path_costs)
        keys = list(cache.keys())
        bank_centers = np.stack([cache_centers[key] for key in keys]).astype(np.float32)
        bank_costs = np.asarray([cache[key] for key in keys], np.float32)
        best = int(np.argmin(bank_costs))
        if bank_costs[best] > start_costs[0] + 1e-5:
            raise AssertionError("strong search violated warm floor")
        start_centers_all.append(starts); start_costs_all.append(start_costs)
        terminal_centers_all.append(terminal_centers); terminal_costs_all.append(terminal_costs)
        path_centers_all.append(state_path_centers); path_costs_all.append(state_path_costs)
        oracle_centers.append(bank_centers[best]); oracle_costs.append(bank_costs[best])
        evaluation_counts.append(len(bank_costs))
        evaluated_centers_rows.append(bank_centers); evaluated_cost_rows.append(bank_costs)
        print(
            f"[{ordinal + 1:03d}/120] {source['episode_id'][row]} "
            f"{float(source['state_six'][row, 3]) * 3.6:.1f}kph "
            f"J={start_costs[0]:.1f}->{bank_costs[best]:.1f} "
            f"budget={len(bank_costs)}", flush=True,
        )

    max_evaluations = max(evaluation_counts)
    evaluated_centers = np.full((120, max_evaluations, 8, 2), np.nan, np.float32)
    evaluated_costs = np.full((120, max_evaluations), np.nan, np.float32)
    for index, (centers, costs) in enumerate(zip(evaluated_centers_rows, evaluated_cost_rows)):
        evaluated_centers[index, : len(costs)] = centers
        evaluated_costs[index, : len(costs)] = costs
    start_centers_all = np.asarray(start_centers_all, np.float32)
    start_costs_all = np.asarray(start_costs_all, np.float32)
    terminal_centers_all = np.asarray(terminal_centers_all, np.float32)
    terminal_costs_all = np.asarray(terminal_costs_all, np.float32)
    path_centers_all = np.asarray(path_centers_all, np.float32)
    path_costs_all = np.asarray(path_costs_all, np.float32)
    oracle_centers = np.asarray(oracle_centers, np.float32)
    oracle_costs = np.asarray(oracle_costs, np.float32)
    warm = start_costs_all[:, 0]
    teacher_cost = start_costs_all[:, 1]
    oracle_gain = warm - oracle_costs
    teacher_gain = warm - teacher_cost

    output.mkdir(parents=True)
    artifact_path = output / "oracle.npz"
    np.savez_compressed(
        artifact_path, source_indices=selected_rows,
        episode_id=source["episode_id"][selected_rows],
        scenario_class=source["scenario_class"][selected_rows],
        nominal_speed_kph=source["speed_kph"][selected_rows],
        control_step=source["control_step"][selected_rows],
        actual_vx_mps=source["state_six"][selected_rows, 3],
        start_names=np.asarray(START_NAMES), start_centers=start_centers_all,
        start_costs=start_costs_all, terminal_centers=terminal_centers_all,
        terminal_costs=terminal_costs_all, path_centers=path_centers_all,
        path_costs=path_costs_all, oracle_centers=oracle_centers,
        oracle_costs=oracle_costs, evaluation_counts=np.asarray(evaluation_counts, np.int32),
        evaluated_centers=evaluated_centers, evaluated_costs=evaluated_costs,
    )
    by_speed = {}
    for speed in sorted(np.unique(source["speed_kph"][selected_rows]).tolist()):
        mask = np.isclose(source["speed_kph"][selected_rows], speed)
        by_speed[str(int(speed))] = {
            "contexts": int(mask.sum()),
            "warm_relative_reduction": float(oracle_gain[mask].sum() / warm[mask].sum()),
            "teacher_headroom_recovery": float(oracle_gain[mask].sum() / teacher_gain[mask].sum()),
        }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_STRONG_SEARCH_ORACLE_TRAIN_ONLY",
        "sources": {
            "replay": str(replay_path), "replay_sha256": sha256(replay_path),
            "teacher": str(teacher_path), "teacher_sha256": sha256(teacher_path),
            "pretrain_summary": str(pretrain_summary_path),
            "pretrain_summary_sha256": sha256(pretrain_summary_path),
            "pretrain_validator_sha256": sha256(pretrain_validator_path),
            "oac_summary": str(oac_summary_path), "oac_summary_sha256": sha256(oac_summary_path),
            "oac_validator_sha256": sha256(oac_validator_path),
        },
        "artifact": str(artifact_path.resolve()), "artifact_sha256": sha256(artifact_path),
        "contract": {
            "split": "train-only; one control_step=0 state from each of four episodes per speed/scenario stratum",
            "contexts": 120, "starts": list(START_NAMES),
            "radii_sigma": list(RADII),
            "directions": "Hadamard/base and two deterministic Givens-rotated full-rank bases",
            "update": "per-start strict best-improvement pattern search; no elite averaging",
            "maximum_candidate_budget_per_state": 965,
            "objective": "deterministic fixed-DBM J50 direct center cost",
            "warm_preserved": True, "dbm_gradient_used": False,
            "formal_validation_or_test_created": False,
        },
        "results": {
            "evaluation_count": distribution(np.asarray(evaluation_counts)),
            "warm_cost": distribution(warm), "teacher_cost": distribution(teacher_cost),
            "oracle_cost": distribution(oracle_costs), "oracle_gain": distribution(oracle_gain),
            "warm_relative_reduction": float(oracle_gain.sum() / warm.sum()),
            "teacher_warm_relative_reduction": float(teacher_gain.sum() / warm.sum()),
            "headroom_multiple_vs_proximal_teacher": float(oracle_gain.sum() / teacher_gain.sum()),
            "baseline_violations": int(np.sum(oracle_costs > warm + 1e-5)),
            "strict_improvement_fraction": float(np.mean(oracle_gain > 1e-5)),
        },
        "by_nominal_speed_kph": by_speed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
