#!/usr/bin/env python3
"""Numerical best-found DBM reference around the final high-speed Actor.

This is an offline audit only.  It deliberately uses exact DBM autograd and
must never be confused with the deployable Actor/Critic contract.  The result
is a multi-start numerical best-found reference, not a certified global
optimum.
"""

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


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1"
)
DEFAULT_OLD_ORACLE = Path(
    "outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v2"
)
DEFAULT_FOLD0 = Path(
    "outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold0_20260831_v1"
)
DEFAULT_FOLD1TO4 = Path(
    "outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold1to4_20260831_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32)
LOW = np.asarray((-1.0, -1.0), np.float32)
HIGH = np.asarray((1.0, 1.0), np.float32)
BASE_START_NAMES = (
    "warm", "proximal_teacher", "old_oracle", "actor_seed0", "actor_seed1", "actor_seed2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--old-oracle-dir", type=Path, default=DEFAULT_OLD_ORACLE)
    parser.add_argument("--actor-fold0-dir", type=Path, default=DEFAULT_FOLD0)
    parser.add_argument("--actor-fold1to4-dir", type=Path, default=DEFAULT_FOLD1TO4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats-per-cell", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--local-random-starts", type=int, default=2)
    parser.add_argument("--global-random-starts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(0.003, 0.01, 0.03))
    parser.add_argument("--seed", type=int, default=26083191)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
        "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def select_rows(source: dict[str, np.ndarray], repeats: int) -> np.ndarray:
    rows: list[int] = []
    for speed in sorted(np.unique(source["speed_kph"]).tolist()):
        for scenario in sorted(np.unique(source["scenario_class"]).astype(str).tolist()):
            mask = (
                np.isclose(source["speed_kph"], speed)
                & (source["scenario_class"].astype(str) == scenario)
                & (source["control_step"] == 0)
            )
            local = np.flatnonzero(mask)
            if len(local) != 4 or len(np.unique(source["episode_id"][local])) != 4:
                raise AssertionError(f"expected four independent rows for {(speed, scenario)}")
            rows.extend(local[:repeats].tolist())
    return np.asarray(rows, np.int64)


def load_final_actor_centers(roots: tuple[Path, Path], count: int) -> np.ndarray:
    centers = np.full((3, count, 8, 2), np.nan, np.float32)
    coverage = np.zeros((3, count), np.int32)
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        for record in summary["records"]:
            fold = int(record["fold"])
            seed = int(record["seed"])
            arm = record["arms"]["16"]
            path = Path(arm["evaluation"])
            if sha256(path) != arm["evaluation_sha256"]:
                raise AssertionError(f"evaluation hash mismatch: fold={fold}, seed={seed}")
            with np.load(path, allow_pickle=False) as loaded:
                rows = np.asarray(loaded["oof_indices"], np.int64)
                centers[seed, rows] = loaded["selected_oof_center"]
                coverage[seed, rows] += 1
    if not np.array_equal(coverage, np.ones_like(coverage)) or not np.isfinite(centers).all():
        raise AssertionError("final Actor OOF centers do not cover every source row exactly once")
    return centers


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    replay_path = (args.replay_dir / "replay.npz").resolve()
    teacher_path = (args.teacher_dir / "labels.npz").resolve()
    old_oracle_path = (args.old_oracle_dir / "oracle.npz").resolve()
    replay_summary = json.loads((args.replay_dir / "summary.json").read_text())
    teacher_summary = json.loads((args.teacher_dir / "summary.json").read_text())
    if replay_summary["formal_validation_or_test_created"]:
        raise AssertionError("replay is not train-only")
    if teacher_summary["protocol"]["formal_validation_or_test_created"]:
        raise AssertionError("teacher is not train-only")
    if sha256(replay_path) != replay_summary["archive_sha256"]:
        raise AssertionError("replay hash mismatch")
    if sha256(teacher_path) != teacher_summary["labels_sha256"]:
        raise AssertionError("teacher hash mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(teacher_path, allow_pickle=False) as loaded:
        teacher = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(old_oracle_path, allow_pickle=False) as loaded:
        old_oracle = {name: np.asarray(loaded[name]) for name in loaded.files}
    count = len(source["state_six"])
    actor_centers = load_final_actor_centers(
        (args.actor_fold0_dir, args.actor_fold1to4_dir), count,
    )
    rows = select_rows(source, args.repeats_per_cell)
    old_lookup = {int(row): index for index, row in enumerate(old_oracle["source_indices"])}
    if any(int(row) not in old_lookup for row in rows):
        raise AssertionError("selected row missing from old oracle")
    old_index = np.asarray([old_lookup[int(row)] for row in rows], np.int64)

    starts = np.stack((
        source["mean_knots_before"][rows], teacher["teacher_knots"][rows],
        old_oracle["oracle_centers"][old_index], actor_centers[0, rows],
        actor_centers[1, rows], actor_centers[2, rows],
    ), axis=1).astype(np.float32)
    start_names = list(BASE_START_NAMES)
    rng = np.random.default_rng(args.seed)
    actor_mean = actor_centers[:, rows].mean(axis=0)
    if args.local_random_starts:
        local = actor_mean[:, None] + rng.normal(
            size=(len(rows), args.local_random_starts, 8, 2)
        ).astype(np.float32) * SIGMA[None, None, None] * 0.75
        starts = np.concatenate((starts, np.clip(local, LOW, HIGH)), axis=1)
        start_names.extend(f"actor_local_random{i}" for i in range(args.local_random_starts))
    if args.global_random_starts:
        global_starts = rng.uniform(
            LOW, HIGH, size=(len(rows), args.global_random_starts, 8, 2)
        ).astype(np.float32)
        starts = np.concatenate((starts, global_starts), axis=1)
        start_names.extend(f"global_random{i}" for i in range(args.global_random_starts))

    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    state = torch.as_tensor(source["state_six"][rows], device=device)
    current_action = torch.as_tensor(source["current_action"][rows], device=device)
    reference = torch.as_tensor(source["reference"][rows, 1:], device=device)
    base = torch.as_tensor(starts, device=device)

    with torch.no_grad():
        initial_cost = batched_cost(
            backend, weights, interpolate_knots(base, params.horizon),
            state, current_action, reference,
        )
    variants: list[torch.nn.Parameter] = []
    variant_names = []
    for learning_rate in args.learning_rates:
        variants.append(torch.nn.Parameter(base.clone()))
        variant_names.extend(f"{name}@lr{learning_rate:g}" for name in start_names)
    optimizer = torch.optim.Adam([
        {"params": [parameter], "lr": float(learning_rate)}
        for parameter, learning_rate in zip(variants, args.learning_rates)
    ])
    center = torch.cat([parameter.detach() for parameter in variants], dim=1)
    repeated_initial_cost = initial_cost.repeat(1, len(args.learning_rates))
    best_cost = repeated_initial_cost.clone()
    best_center = center.detach().clone()
    trace_steps = sorted(set((0, 10, 25, 50, 100, 200, 400, 800, args.steps)))
    trace_steps = [step for step in trace_steps if step <= args.steps]
    trace_cost, trace_variant = [], []
    warm_cost = initial_cost[:, 0].detach()

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        center = torch.cat(variants, dim=1)
        actions = interpolate_knots(center, params.horizon)
        cost = batched_cost(backend, weights, actions, state, current_action, reference)
        improved = torch.isfinite(cost) & (cost < best_cost)
        best_cost = torch.where(improved, cost.detach(), best_cost)
        best_center = torch.where(improved[..., None, None], center.detach(), best_center)
        if step in trace_steps:
            per_state, which = best_cost.min(dim=1)
            trace_cost.append(per_state.cpu().numpy())
            trace_variant.append(which.cpu().numpy())
            print(
                f"step={step:04d} mean_best={float(per_state.mean()):.6f} "
                f"median_best={float(per_state.median()):.6f}", flush=True,
            )
        if step == args.steps:
            break
        # Each state/variant is an independent optimization problem.  Scaling
        # by warm cost keeps high-speed magnitudes from dominating numerics.
        (cost / warm_cost[:, None]).mean().backward()
        if any(parameter.grad is None for parameter in variants):
            raise AssertionError("missing action gradient")
        torch.nn.utils.clip_grad_norm_(variants, 10.0)
        optimizer.step()
        with torch.no_grad():
            for parameter in variants:
                parameter.clamp_(-1.0, 1.0)

    per_state_cost, best_variant = best_cost.min(dim=1)
    row_index = torch.arange(len(rows), device=device)
    per_state_center = best_center[row_index, best_variant]
    actor_costs = initial_cost[:, 3:6]
    actor_mean_cost = actor_costs.mean(dim=1)
    actor_best_cost = actor_costs.min(dim=1).values
    gap_fixed_actor = actor_mean_cost - per_state_cost
    gap_best_actor = actor_best_cost - per_state_cost
    by_speed = {}
    for speed in sorted(np.unique(source["speed_kph"][rows]).tolist()):
        mask_np = np.isclose(source["speed_kph"][rows], speed)
        mask = torch.as_tensor(mask_np, device=device)
        by_speed[str(int(speed))] = {
            "states": int(mask.sum()),
            "fixed_actor_mean_cost": float(actor_mean_cost[mask].mean()),
            "best_found_mean_cost": float(per_state_cost[mask].mean()),
            "aggregate_gap_fraction_of_actor": float(
                gap_fixed_actor[mask].sum() / actor_mean_cost[mask].sum()
            ),
            "paired_gap_fraction_median": float(
                torch.median(gap_fixed_actor[mask] / actor_mean_cost[mask])
            ),
        }
    summary = {
        "format": "highspeed_final_actor_numerical_oracle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "NUMERICAL_BEST_FOUND_NOT_CERTIFIED_GLOBAL_OPTIMUM",
        "contract": {
            "objective": "deterministic DBM J50 in the physical 16D uniform-knot box",
            "offline_dbm_autograd_only": True,
            "deployable_or_query_compatible": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "states": len(rows), "repeats_per_speed_scenario_cell": args.repeats_per_cell,
            "base_starts": start_names, "learning_rates": args.learning_rates,
            "steps": args.steps, "trace_steps": trace_steps,
            "physical_action_bounds": [LOW.tolist(), HIGH.tolist()],
        },
        "results": {
            "warm_cost": distribution(initial_cost[:, 0].cpu().numpy()),
            "old_oracle_cost": distribution(initial_cost[:, 2].cpu().numpy()),
            "fixed_actor_three_seed_cost": distribution(actor_costs.cpu().numpy()),
            "fixed_actor_per_state_seed_mean_cost": distribution(actor_mean_cost.cpu().numpy()),
            "fixed_actor_per_state_best_seed_cost": distribution(actor_best_cost.cpu().numpy()),
            "numerical_best_found_cost": distribution(per_state_cost.cpu().numpy()),
            "fixed_actor_gap_to_best_found": distribution(gap_fixed_actor.cpu().numpy()),
            "fixed_actor_aggregate_gap_fraction": float(gap_fixed_actor.sum() / actor_mean_cost.sum()),
            "fixed_actor_paired_gap_fraction": distribution(
                (gap_fixed_actor / actor_mean_cost).cpu().numpy()
            ),
            "best_of_three_actor_aggregate_gap_fraction": float(
                gap_best_actor.sum() / actor_best_cost.sum()
            ),
            "best_found_beats_fixed_actor_fraction": float(
                (per_state_cost[:, None] < actor_costs - 1e-4).float().mean()
            ),
            "by_speed_kph": by_speed,
        },
        "inputs": {
            "replay": str(replay_path), "replay_sha256": sha256(replay_path),
            "teacher": str(teacher_path), "teacher_sha256": sha256(teacher_path),
            "old_oracle": str(old_oracle_path), "old_oracle_sha256": sha256(old_oracle_path),
        },
    }
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "solutions.npz", source_indices=rows,
        episode_id=source["episode_id"][rows], scenario_class=source["scenario_class"][rows],
        speed_kph=source["speed_kph"][rows], start_names=np.asarray(start_names),
        starts=starts, initial_cost=initial_cost.cpu().numpy(),
        variant_names=np.asarray(variant_names), best_variant=best_variant.cpu().numpy(),
        best_center=per_state_center.cpu().numpy(), best_cost=per_state_cost.cpu().numpy(),
        trace_steps=np.asarray(trace_steps), trace_best_cost=np.asarray(trace_cost),
        trace_best_variant=np.asarray(trace_variant),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
