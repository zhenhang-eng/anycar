#!/usr/bin/env python3
"""Generate train-only proximal-search teachers for high-speed DBM contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), dtype=np.float32)
ACTION_MIN = np.asarray((-1.0, -1.0), dtype=np.float32)
ACTION_MAX = np.asarray((1.0, 1.0), dtype=np.float32)
ELITE_COUNT = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-chunk", type=int, default=256)
    return parser.parse_args()


def ring(base: np.ndarray, radius: float, directions: np.ndarray) -> np.ndarray:
    delta = radius * SIGMA.reshape(1, 1, 2) * directions
    return np.concatenate((base[None] + delta, base[None] - delta), axis=0).astype(
        np.float32
    )


def elite_mean(centers: np.ndarray, costs: np.ndarray) -> np.ndarray:
    elite = np.argsort(costs)[: min(ELITE_COUNT, len(costs))]
    return centers[elite].mean(axis=0).astype(np.float32)


def metric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    replay_dir = args.replay_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    replay_path = replay_dir / "replay.npz"
    replay_summary_path = replay_dir / "summary.json"
    replay_summary = json.loads(replay_summary_path.read_text())
    if replay_summary["formal_validation_or_test_created"]:
        raise AssertionError("replay is not train-only")
    if sha256(replay_path) != replay_summary["archive_sha256"]:
        raise AssertionError("source replay hash mismatch")

    source = np.load(replay_path, allow_pickle=False)
    count = len(source["state_six"])
    directions = hadamard_directions().astype(np.float32)
    params = TorchMPPIParams(num_samples=64)
    weights = TorchMPPICostWeights()
    backend = TorchDynamicBicycleRolloutBackend()
    device = torch.device(args.device)

    all_centers = []
    all_costs = []
    teacher = []
    teacher_cost = []
    anchor_cost = []
    gains = []
    stays = []
    clip_fractions = []
    for index in range(count):
        state = torch.as_tensor(
            source["state_six"][index : index + 1], dtype=torch.float32, device=device
        )
        current = torch.as_tensor(
            source["current_action"][index : index + 1],
            dtype=torch.float32,
            device=device,
        )
        reference_np = source["reference"][index]
        if len(reference_np) == params.horizon + 1:
            reference_np = reference_np[1:]
        reference = torch.as_tensor(
            reference_np[None], dtype=torch.float32, device=device
        )
        cache: dict[bytes, float] = {}

        def evaluate(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            clipped = np.clip(candidates, ACTION_MIN, ACTION_MAX).astype(np.float32)
            keys = [np.round(row, 7).tobytes() for row in clipped]
            pending = []
            pending_keys = []
            for key, value in zip(keys, clipped):
                if key not in cache and key not in pending_keys:
                    pending_keys.append(key)
                    pending.append(value)
            if pending:
                pending_array = np.asarray(pending, dtype=np.float32)
                values = []
                with torch.no_grad():
                    for start in range(0, len(pending_array), args.eval_chunk):
                        knots = torch.as_tensor(
                            pending_array[start : start + args.eval_chunk][None],
                            dtype=torch.float32,
                            device=device,
                        )
                        actions = interpolate_knots(knots, params.horizon)
                        values.append(
                            batched_cost(
                                backend, weights, actions, state, current, reference
                            )[0].cpu().numpy()
                        )
                for key, value in zip(pending_keys, np.concatenate(values)):
                    cache[key] = float(value)
            return clipped, np.asarray([cache[key] for key in keys], np.float32)

        anchor = np.asarray(source["mean_knots_before"][index], np.float32)
        anchor_clipped, anchor_value = evaluate(anchor[None])
        first_raw = np.concatenate(
            (ring(anchor, 0.25, directions), ring(anchor, 0.50, directions)), axis=0
        )
        first, first_cost = evaluate(first_raw)
        mean1 = elite_mean(
            np.concatenate((anchor_clipped, first)),
            np.concatenate((anchor_value, first_cost)),
        )
        second, second_cost = evaluate(ring(mean1, 0.50, directions))
        mean2 = elite_mean(
            np.concatenate((anchor_clipped, first, second)),
            np.concatenate((anchor_value, first_cost, second_cost)),
        )
        third, third_cost = evaluate(ring(mean2, 0.25, directions))
        centers = np.concatenate((anchor_clipped, first, second, third))
        costs = np.concatenate((anchor_value, first_cost, second_cost, third_cost))
        best = int(np.argmin(costs))
        best_center = centers[best]
        best_cost = float(costs[best])
        gain = float(costs[0] - best_cost)
        stay = gain <= 1e-6
        if best_cost > float(costs[0]) + 1e-6:
            raise AssertionError("warm-preserving search regressed")
        all_centers.append(centers)
        all_costs.append(costs)
        teacher.append(anchor if stay else best_center)
        teacher_cost.append(best_cost)
        anchor_cost.append(float(costs[0]))
        gains.append(gain)
        stays.append(stay)
        raw = np.concatenate((anchor[None], first_raw, ring(mean1, 0.50, directions), ring(mean2, 0.25, directions)))
        clip_fractions.append(float(np.mean(raw != centers)))
        print(
            f"[{index + 1:03d}/{count:03d}] "
            f"{source['episode_id'][index]}#{int(source['control_step'][index])} "
            f"vx={float(source['state_six'][index, 3]) * 3.6:.1f}kph "
            f"J={float(costs[0]):.1f}->{best_cost:.1f} gain={gain:.1f}",
            flush=True,
        )

    anchor_cost = np.asarray(anchor_cost, np.float32)
    teacher_cost = np.asarray(teacher_cost, np.float32)
    gains = np.asarray(gains, np.float32)
    stays = np.asarray(stays, bool)
    actual_kph = source["state_six"][:, 3] * 3.6
    target_mask = (actual_kph >= 40.0) & (actual_kph <= 100.0)
    episodes = np.asarray(source["episode_id"])
    unique_episodes = np.unique(episodes)
    rng = np.random.default_rng(20260830)
    bootstrap = []
    for _ in range(2000):
        sampled = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        indices = np.concatenate([np.flatnonzero(episodes == item) for item in sampled])
        bootstrap.append(float(gains[indices].sum() / anchor_cost[indices].sum()))

    by_nominal_speed = {}
    for speed in np.unique(source["speed_kph"]):
        mask = np.isclose(source["speed_kph"], speed)
        by_nominal_speed[str(int(speed))] = {
            "contexts": int(mask.sum()),
            "actual_vx_kph": metric(actual_kph[mask]),
            "anchor_cost": metric(anchor_cost[mask]),
            "teacher_cost": metric(teacher_cost[mask]),
            "gain": metric(gains[mask]),
            "stay_fraction": float(stays[mask].mean()),
        }

    output.mkdir(parents=True)
    labels_path = output / "labels.npz"
    np.savez_compressed(
        labels_path,
        episode_id=source["episode_id"],
        scenario_class=source["scenario_class"],
        nominal_speed_kph=source["speed_kph"],
        control_step=source["control_step"],
        actual_vx_mps=source["state_six"][:, 3],
        target_speed_mask=target_mask,
        anchor_knots=source["mean_knots_before"],
        teacher_knots=np.asarray(teacher, np.float32),
        anchor_cost=anchor_cost,
        teacher_cost=teacher_cost,
        gain=gains,
        stay=stays,
        search_centers=np.asarray(all_centers, np.float32),
        search_costs=np.asarray(all_costs, np.float32),
        clip_fraction=np.asarray(clip_fractions, np.float32),
    )
    bootstrap = np.asarray(bootstrap)
    summary = {
        "qualification": "HIGHSPEED_PROXIMAL_SEARCH_TEACHER_COMPLETE_TRAIN_ONLY",
        "source_replay": str(replay_path),
        "source_replay_sha256": sha256(replay_path),
        "labels": str(labels_path),
        "labels_sha256": sha256(labels_path),
        "protocol": {
            "objective": "deterministic fixed-DBM J50 direct center cost",
            "anchor": "trace warm mean_knots_before; no Actor checkpoint",
            "candidate_budget_per_state": 129,
            "rounds": "anchor + 64 ring + 32 recentered + 32 recentered",
            "sigma": SIGMA.tolist(),
            "warm_preserved": True,
            "formal_validation_or_test_created": False,
        },
        "results": {
            "contexts": count,
            "candidate_center_rollouts": count * 129,
            "actual_target_speed_contexts": int(target_mask.sum()),
            "actual_target_speed_fraction": float(target_mask.mean()),
            "anchor_cost": metric(anchor_cost),
            "teacher_cost": metric(teacher_cost),
            "gain": metric(gains),
            "aggregate_relative_cost_reduction": float(
                gains.sum() / anchor_cost.sum()
            ),
            "episode_bootstrap_ci": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "strict_improvement_fraction": float((gains > 1e-6).mean()),
            "stay_fraction": float(stays.mean()),
            "baseline_violations": int(np.sum(teacher_cost > anchor_cost + 1e-6)),
            "clip_fraction": metric(np.asarray(clip_fractions)),
        },
        "by_nominal_speed_kph": by_nominal_speed,
        "notes": [
            "Teacher quality is measured only relative to the warm center.",
            "No J16/global optimum is claimed or required for this cross-domain pilot.",
            "The complete 129-center bank is retained for later ranking or distillation diagnostics.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
