#!/usr/bin/env python3
"""G0 reward-surface smoothness re-analysis (pure derived analysis, no rollout).

Implements §11.1 of car_foundation/docs/mppi_sampling_center_review_20260812.md.

It re-analyzes two frozen, immutable, train-only sidecars and answers three
questions about the reward landscape ``reward = J_direct(anchor) - J_direct(center)``
without running a single new DBM rollout:

  D1. Intra-state quadratic fit quality -- per frame, fit a diagonal quadratic in
      the 16 orthonormal Hadamard directions and report R^2.  Low R^2 means the
      local cost surface is not well described by a quadratic at the radii the
      critic uses, i.e. the gradient the critic is asked to learn is poorly defined.

  D3. Finite-difference radius sensitivity -- the sidecar stores directional slopes
      at two radii (0.05 sigma and 0.15 sigma).  If the slope sign flips between
      radii, the gradient direction is not stable at the critic's operating scale.

  D2. Alpha-line shape -- the trust-region sidecar stores a 21-point cost line
      ``alpha in 0:0.05:1`` per context.  Count strict interior local minima and the
      fraction of negative discrete second differences, stratified by reference
      speed.  A multi-modal or negatively-curved line means the warm->proposal
      direction itself is not a clean descent.

Everything is stratified by reference speed (1.2/1.6/2.0/2.4/2.8 m/s) because the
deployed failure mode is concentrated at 2.4/2.8 m/s.

The script loads npz with numpy only.  It does not import torch, does not run the
DBM, and does not touch validation or the sealed test split.  It additionally
independently recomputes the stored directional slopes / curvatures and the line
argmin from raw centers+costs and reports the maximum absolute error (expected ~0)
as the provenance check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LABEL_ROOT = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels")

CURVATURE_SIDECARS = (
    LABEL_ROOT / "dbm_j16_local_curvature_train_diverse_20260807_v1",
    LABEL_ROOT / "dbm_j16_local_curvature_train_expansion_20260807_v1",
)
TRUST_REGION_SIDECAR = (
    LABEL_ROOT / "dbm_direct_trust_region_train_20260807_v1"
)

DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/reward_surface_g0_20260813_v1/analysis.json"
)

SPEED_BINS = (1.2, 1.6, 2.0, 2.4, 2.8)
DIRECTION_COUNT = 16
RADIUS_COUNT = 2
SECOND_DIFF_TOL = 0.0  # a second difference below this counts as non-convex


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speed_bin(speed: float) -> float:
    return min(SPEED_BINS, key=lambda candidate: abs(candidate - speed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="If >0, cap the number of curvature frames analysed (smoke only).",
    )
    return parser.parse_args()


def load_speed_map(source_dir: Path) -> dict[str, float]:
    plan_path = source_dir / "scenario_plan.json"
    plan = json.loads(plan_path.read_text())
    return {
        episode["episode_id"]: float(episode["reference_speed_mps"])
        for episode in plan["episodes"]
    }


def _quad_residual_r2(
    signed_offsets: np.ndarray, values: np.ndarray, base_cost: float
) -> float:
    """Least-squares fit of y = base + g*s + 0.5*h*s^2; return R^2."""
    centered = values - base_cost
    design = np.stack([signed_offsets, 0.5 * signed_offsets ** 2], axis=1)
    solution, *_ = np.linalg.lstsq(design, centered, rcond=None)
    predicted = design @ solution
    ss_residual = float(np.sum((centered - predicted) ** 2))
    ss_total = float(np.sum((centered - centered.mean()) ** 2))
    if ss_total <= 1e-12:
        return 1.0 if ss_residual <= 1e-12 else 0.0
    return 1.0 - ss_residual / ss_total


def analyze_curvature_frame(
    data: np.lib.npyio.NpzFile,
) -> tuple[float, float, float, float]:
    """Return (dir_r2_mean, fd_sign_agree, slope_replay_err, curv_replay_err)."""
    cost = np.asarray(data["direct_cost"], np.float64)
    radii = np.asarray(data["radii_sigma"], np.float64)
    symmetry = np.asarray(data["symmetric_pair_mask"])
    stored_slope = np.asarray(data["directional_slope"], np.float64)
    stored_curv = np.asarray(data["directional_curvature"], np.float64)
    base_cost = float(cost[0])

    replay_slope = np.zeros((RADIUS_COUNT, DIRECTION_COUNT), np.float64)
    replay_curv = np.zeros((RADIUS_COUNT, DIRECTION_COUNT), np.float64)
    for radius_index in range(RADIUS_COUNT):
        radius = float(radii[radius_index])
        start = 1 + radius_index * (2 * DIRECTION_COUNT)
        for direction in range(DIRECTION_COUNT):
            positive = start + 2 * direction
            negative = positive + 1
            cost_positive = float(cost[positive])
            cost_negative = float(cost[negative])
            replay_slope[radius_index, direction] = (
                cost_positive - cost_negative
            ) / (2.0 * radius)
            replay_curv[radius_index, direction] = (
                cost_positive + cost_negative - 2.0 * base_cost
            ) / (radius * radius)

    direction_r2 = []
    sign_matches = 0
    sign_eligible = 0
    for direction in range(DIRECTION_COUNT):
        signed_offsets = []
        values = []
        symmetric_both = True
        for radius_index in range(RADIUS_COUNT):
            radius = float(radii[radius_index])
            start = 1 + radius_index * (2 * DIRECTION_COUNT)
            positive = start + 2 * direction
            negative = positive + 1
            if not bool(symmetry[radius_index, direction]):
                symmetric_both = False
            signed_offsets.extend((radius, -radius))
            values.extend((float(cost[positive]), float(cost[negative])))
        direction_r2.append(
            _quad_residual_r2(
                np.asarray(signed_offsets, np.float64),
                np.asarray(values, np.float64),
                base_cost,
            )
        )
        if symmetric_both:
            sign_eligible += 1
            if replay_slope[0, direction] * replay_slope[1, direction] > 0.0:
                sign_matches += 1

    slope_err = float(np.max(np.abs(replay_slope - stored_slope)))
    curv_err = float(np.max(np.abs(replay_curv - stored_curv)))
    fd_sign_agree = (sign_matches / sign_eligible) if sign_eligible else float("nan")
    return (
        float(np.mean(direction_r2)),
        float(fd_sign_agree),
        slope_err,
        curv_err,
    )


def analyze_alpha_line(data: np.lib.npyio.NpzFile) -> list[dict]:
    direct_cost = np.asarray(data["direct_cost"], np.float64)
    stored_argmin = np.asarray(data["argmin_index"])
    speed = float(data["reference_speed_mps"])
    rows = []
    for context in range(direct_cost.shape[0]):
        line = direct_cost[context]
        interior = line[1:-1]
        local_minima = int(np.sum(
            (interior < line[:-2]) & (interior < line[2:])
        ))
        second_diff = line[2:] - 2.0 * line[1:-1] + line[:-2]
        negative_second_diff_fraction = float(
            np.mean(second_diff < SECOND_DIFF_TOL)
        )
        rows.append(
            {
                "speed": speed_bin(speed),
                "local_minima": local_minima,
                "unimodal": local_minima <= 1,
                "negative_second_diff_fraction": negative_second_diff_fraction,
                "argmin_replay_ok": bool(
                    int(np.argmin(line)) == int(stored_argmin[context])
                ),
            }
        )
    return rows


def summarize_by_speed(records: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["speed"]].append(record[key])
    summary = {}
    for speed in SPEED_BINS:
        values = np.asarray(grouped.get(speed, []), np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[f"{speed}"] = {"n": 0}
            continue
        summary[f"{speed}"] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
        }
    overall = np.asarray([r[key] for r in records], np.float64)
    overall = overall[np.isfinite(overall)]
    summary["overall"] = {
        "n": int(overall.size),
        "mean": float(overall.mean()) if overall.size else float("nan"),
    }
    return summary


def main() -> None:
    args = parse_args()

    inputs = {
        "trust_region_summary": str(TRUST_REGION_SIDECAR / "summary.json"),
        "trust_region_config": str(TRUST_REGION_SIDECAR / "config.json"),
    }
    for index, sidecar in enumerate(CURVATURE_SIDECARS):
        inputs[f"curvature_summary_{index}"] = str(sidecar / "summary.json")

    input_hashes = {
        name: sha256_file(Path(path)) for name, path in inputs.items()
    }

    # --- D1 / D3: curvature sidecars -------------------------------------
    curvature_records = []
    max_slope_replay_error = 0.0
    max_curv_replay_error = 0.0
    max_base_replay_error = 0.0
    frame_count = 0
    for sidecar in CURVATURE_SIDECARS:
        summary = json.loads((sidecar / "summary.json").read_text())
        speed_map = load_speed_map(Path(summary["source"]))
        for episode_dir in sorted(sidecar.iterdir()):
            if not episode_dir.is_dir() or not episode_dir.name.startswith("episode"):
                continue
            speed = speed_map.get(episode_dir.name)
            if speed is None:
                continue
            for step_file in sorted(episode_dir.glob("step_*.npz")):
                if args.max_frames and frame_count >= args.max_frames:
                    break
                data = np.load(step_file, allow_pickle=False)
                base_replay = float(abs(
                    float(data["direct_cost"][0]) - float(data["base_j16_cost"])
                ))
                max_base_replay_error = max(max_base_replay_error, base_replay)
                dir_r2, fd_sign, slope_err, curv_err = analyze_curvature_frame(data)
                max_slope_replay_error = max(max_slope_replay_error, slope_err)
                max_curv_replay_error = max(max_curv_replay_error, curv_err)
                curvature_records.append(
                    {
                        "speed": speed_bin(speed),
                        "direction_r2_mean": dir_r2,
                        "fd_sign_agreement": fd_sign,
                    }
                )
                frame_count += 1

    # --- D2: alpha-line sidecar ------------------------------------------
    alpha_records = []
    argmin_replay_failures = 0
    for episode_dir in sorted(TRUST_REGION_SIDECAR.iterdir()):
        if not episode_dir.is_dir() or not episode_dir.name.startswith("episode"):
            continue
        for step_file in sorted(episode_dir.glob("step_*.npz")):
            data = np.load(step_file, allow_pickle=False)
            for row in analyze_alpha_line(data):
                if not row["argmin_replay_ok"]:
                    argmin_replay_failures += 1
                alpha_records.append(row)

    result = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "recomputes_dynamics": False,
        "qualification": (
            "TRAIN_ONLY_MECHANISM_DIAGNOSTIC; validation and sealed test not "
            "loaded; no new rollout executed"
        ),
        "inputs": inputs,
        "input_sha256": input_hashes,
        "curvature_frame_count": frame_count,
        "alpha_line_record_count": len(alpha_records),
        "independent_recomputation": {
            "base_cost_replay_max_abs_error": max_base_replay_error,
            "directional_slope_replay_max_abs_error": max_slope_replay_error,
            "directional_curvature_replay_max_abs_error": max_curv_replay_error,
            "alpha_argmin_replay_failures": argmin_replay_failures,
        },
        "diagnostics": {
            "D1_direction_quadratic_r2_by_speed": summarize_by_speed(
                curvature_records, "direction_r2_mean"
            ),
            "D3_fd_sign_agreement_by_speed": summarize_by_speed(
                curvature_records, "fd_sign_agreement"
            ),
            "D2_alpha_local_minima_by_speed": summarize_by_speed(
                alpha_records, "local_minima"
            ),
            "D2_alpha_negative_second_diff_fraction_by_speed": summarize_by_speed(
                alpha_records, "negative_second_diff_fraction"
            ),
        },
        "caveats": [
            "D1 uses a diagonal quadratic in the Hadamard basis; cross-direction "
            "curvature is not estimable from axis-aligned antithetic probes.",
            "D3 sign agreement only counts directions whose antithetic pair is "
            "symmetric at BOTH radii (no clip asymmetry).",
            "Speeds are binned to the nearest of 1.2/1.6/2.0/2.4/2.8 m/s.",
            "This is a train-only diagnostic; it says nothing about generalization.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"curvature frames analysed: {frame_count}")
    print(f"alpha-line records: {len(alpha_records)}")
    print(
        "replay max err  base="
        f"{max_base_replay_error:.3e} slope={max_slope_replay_error:.3e} "
        f"curv={max_curv_replay_error:.3e} argmin_fail={argmin_replay_failures}"
    )
    for name, block in result["diagnostics"].items():
        print(f"\n{name}")
        for speed, stats in block.items():
            if stats.get("n", 0) == 0:
                continue
            print(
                f"  {speed:>8}: n={stats['n']:<6} mean={stats['mean']:.4f}"
                + (
                    f" median={stats['median']:.4f} p05={stats['p05']:.4f}"
                    if "median" in stats
                    else ""
                )
            )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
