#!/usr/bin/env python3
"""G0.1 cross-radius reward-surface analysis (pure derived analysis, no rollout).

Implements the G0.1 step of §11.7 in
car_foundation/docs/mppi_sampling_center_review_20260812.md.  It addresses the
review's objections to G0 by turning the in-sample R^2 into a genuine
cross-radius *prediction* error, replacing the crude FD sign-agreement with
gradient-vector cosine / magnitude / top-k metrics, separating clipped from
unclipped frames, and tightening the alpha-line diagnostics.

Everything is computed from the frozen train-only sidecars with numpy only; no
DBM rollout is executed and no torch is imported.

Curvature sidecar diagnostics (per frame, 16 Hadamard directions, radii 0.05/0.15):
  C1. Cross-radius prediction error -- fit per-direction slope+curvature from the
      inner radius points and predict the outer radius costs (and the reverse),
      reporting absolute and cost-normalized error.  This is an extrapolation
      error, unlike the in-sample R^2.
  C2. Gradient-vector stability -- cosine(g_0.05, g_0.15), magnitude ratio
      ||g_0.15||/||g_0.05||, and top-k dominant-direction sign agreement.
  C3. Clipping stratification -- every metric is reported separately for frames
      with at least one clipped antithetic pair versus fully symmetric frames.

Alpha-line sidecar diagnostics (tightened D2):
  A1. Fraction of lines with any negative discrete second difference.
  A2. argmin location: alpha=0 / interior / alpha=1.
  A3. Fraction of monotone non-increasing (clean-descent) lines.
  A4. Max negative curvature magnitude and its tail.

All aggregates are stratified by reference speed.
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
    "outputs/mppi_proposal/reward_surface_g0_1_20260813_v1/analysis.json"
)

SPEED_BINS = (1.2, 1.6, 2.0, 2.4, 2.8)
DIRECTION_COUNT = 16
RADIUS_COUNT = 2
TOP_K = 4
ZERO_GRAD_NORM = 1e-8


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
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_speed_map(source_dir: Path) -> dict[str, float]:
    plan = json.loads((source_dir / "scenario_plan.json").read_text())
    return {
        episode["episode_id"]: float(episode["reference_speed_mps"])
        for episode in plan["episodes"]
    }


def _direction_costs(cost, radii):
    """Return per-radius, per-direction (positive, negative) costs."""
    pairs = []
    for radius_index in range(RADIUS_COUNT):
        start = 1 + radius_index * (2 * DIRECTION_COUNT)
        radius_pairs = []
        for direction in range(DIRECTION_COUNT):
            positive = start + 2 * direction
            radius_pairs.append((float(cost[positive]), float(cost[positive + 1])))
        pairs.append(radius_pairs)
    return pairs


def cross_radius_prediction_error(cost, radii):
    """Fit inner radius slope+curvature, predict outer radius; return abs errors."""
    base = float(cost[0])
    pairs = _direction_costs(cost, radii)
    errors_forward = []  # predict 0.15 from 0.05
    errors_backward = []  # predict 0.05 from 0.15
    for direction in range(DIRECTION_COUNT):
        cp_in, cn_in = pairs[0][direction]
        cp_out, cn_out = pairs[1][direction]
        r_in, r_out = float(radii[0]), float(radii[1])
        # Fit from inner.
        slope_in = (cp_in - cn_in) / (2.0 * r_in)
        curv_in = (cp_in + cn_in - 2.0 * base) / (r_in * r_in)
        pred_cp_out = base + slope_in * r_out + 0.5 * curv_in * r_out * r_out
        pred_cn_out = base - slope_in * r_out + 0.5 * curv_in * r_out * r_out
        errors_forward.extend((abs(pred_cp_out - cp_out), abs(pred_cn_out - cn_out)))
        # Fit from outer, predict inner.
        slope_out = (cp_out - cn_out) / (2.0 * r_out)
        curv_out = (cp_out + cn_out - 2.0 * base) / (r_out * r_out)
        pred_cp_in = base + slope_out * r_in + 0.5 * curv_out * r_in * r_in
        pred_cn_in = base - slope_out * r_in + 0.5 * curv_out * r_in * r_in
        errors_backward.extend((abs(pred_cp_in - cp_in), abs(pred_cn_in - cn_in)))
    return (
        float(np.mean(errors_forward)),
        float(np.mean(errors_backward)),
        float(np.max(errors_forward)),
        float(base),
    )


def gradient_stability(cost, radii, symmetry):
    """Return (cosine, magnitude_ratio, topk_agreement, has_valid_grad)."""
    slopes = np.zeros((RADIUS_COUNT, DIRECTION_COUNT), np.float64)
    for radius_index in range(RADIUS_COUNT):
        radius = float(radii[radius_index])
        start = 1 + radius_index * (2 * DIRECTION_COUNT)
        for direction in range(DIRECTION_COUNT):
            positive = start + 2 * direction
            slopes[radius_index, direction] = (
                float(cost[positive]) - float(cost[positive + 1])
            ) / (2.0 * radius)
    g_small = slopes[0]
    g_large = slopes[1]
    norm_small = float(np.linalg.norm(g_small))
    norm_large = float(np.linalg.norm(g_large))
    if norm_small < ZERO_GRAD_NORM or norm_large < ZERO_GRAD_NORM:
        return (float("nan"), float("nan"), float("nan"), False)
    cosine = float(np.dot(g_small, g_large) / (norm_small * norm_large))
    magnitude_ratio = norm_large / norm_small
    top_indices = np.argsort(-np.abs(g_small))[:TOP_K]
    agreements = [
        (g_small[i] * g_large[i]) > 0.0 for i in top_indices if abs(g_small[i]) > 0
    ]
    topk = float(np.mean(agreements)) if agreements else float("nan")
    return (cosine, magnitude_ratio, topk, True)


def analyze_curvature_frame(data):
    cost = np.asarray(data["direct_cost"], np.float64)
    radii = np.asarray(data["radii_sigma"], np.float64)
    symmetry = np.asarray(data["symmetric_pair_mask"])
    has_clip = bool(not symmetry.all())
    err_fwd, err_bwd, err_max, base = cross_radius_prediction_error(cost, radii)
    cosine, mag_ratio, topk, valid_grad = gradient_stability(cost, radii, symmetry)
    return {
        "clipped": has_clip,
        "pred_err_forward": err_fwd,
        "pred_err_backward": err_bwd,
        "pred_err_forward_max": err_max,
        "cost_scale": abs(base) if abs(base) > 1e-6 else 1.0,
        "cosine": cosine,
        "magnitude_ratio": mag_ratio,
        "topk_agreement": topk,
        "valid_grad": valid_grad,
    }


def analyze_alpha_line(data):
    direct_cost = np.asarray(data["direct_cost"], np.float64)
    speed = float(data["reference_speed_mps"])
    rows = []
    for context in range(direct_cost.shape[0]):
        line = direct_cost[context]
        second_diff = line[2:] - 2.0 * line[1:-1] + line[:-2]
        argmin_index = int(np.argmin(line))
        interior = line[1:-1]
        local_minima = int(np.sum((interior < line[:-2]) & (interior < line[2:])))
        monotone_descent = bool(np.all(np.diff(line) <= 1e-9))
        rows.append(
            {
                "speed": speed_bin(speed),
                "any_negative_second_diff": bool(np.any(second_diff < 0.0)),
                "argmin_location": (
                    "alpha0" if argmin_index == 0
                    else "alpha1" if argmin_index == len(line) - 1
                    else "interior"
                ),
                "monotone_descent": monotone_descent,
                "local_minima": local_minima,
                "max_negative_curvature": float(
                    -np.min(second_diff) if np.any(second_diff < 0) else 0.0
                ),
            }
        )
    return rows


def summarize(records, key, speed_key="speed"):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[speed_key]].append(record[key])
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
            "p95": float(np.percentile(values, 95)),
        }
    overall = np.asarray([r[key] for r in records], np.float64)
    overall = overall[np.isfinite(overall)]
    summary["overall"] = {
        "n": int(overall.size),
        "mean": float(overall.mean()) if overall.size else float("nan"),
    }
    return summary


def summarize_fraction(records, key, speed_key="speed"):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[speed_key]].append(bool(record[key]))
    summary = {}
    for speed in SPEED_BINS:
        values = grouped.get(speed, [])
        if not values:
            summary[f"{speed}"] = {"n": 0}
            continue
        summary[f"{speed}"] = {
            "n": len(values),
            "fraction": float(np.mean(values)),
        }
    all_values = [bool(r[key]) for r in records]
    summary["overall"] = {
        "n": len(all_values),
        "fraction": float(np.mean(all_values)) if all_values else float("nan"),
    }
    return summary


def main() -> None:
    args = parse_args()
    inputs = {
        "trust_region_summary": str(TRUST_REGION_SIDECAR / "summary.json"),
    }
    for index, sidecar in enumerate(CURVATURE_SIDECARS):
        inputs[f"curvature_summary_{index}"] = str(sidecar / "summary.json")
    input_hashes = {name: sha256_file(Path(p)) for name, p in inputs.items()}

    curvature_records = []
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
                record = analyze_curvature_frame(np.load(step_file, allow_pickle=False))
                record["speed"] = speed_bin(speed)
                curvature_records.append(record)
                frame_count += 1

    alpha_records = []
    for episode_dir in sorted(TRUST_REGION_SIDECAR.iterdir()):
        if not episode_dir.is_dir() or not episode_dir.name.startswith("episode"):
            continue
        for step_file in sorted(episode_dir.glob("step_*.npz")):
            alpha_records.extend(
                analyze_alpha_line(np.load(step_file, allow_pickle=False))
            )

    clipped_records = [r for r in curvature_records if r["clipped"]]
    unclipped_records = [r for r in curvature_records if not r["clipped"]]
    valid_grad_records = [r for r in curvature_records if r["valid_grad"]]

    def norm_pred_err(records):
        out = []
        for r in records:
            out.append({**r, "pred_err_forward_norm": r["pred_err_forward"] / r["cost_scale"]})
        return out

    result = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "recomputes_dynamics": False,
        "qualification": (
            "TRAIN_ONLY_MECHANISM_DIAGNOSTIC; algebraic reconstruction from stored "
            "costs only, no new DBM rollout; validation and sealed test not loaded"
        ),
        "inputs": inputs,
        "input_sha256": input_hashes,
        "curvature_frame_count": frame_count,
        "clipped_frame_count": len(clipped_records),
        "unclipped_frame_count": len(unclipped_records),
        "alpha_line_record_count": len(alpha_records),
        "diagnostics": {
            "C1_cross_radius_pred_err_forward_by_speed": summarize(
                curvature_records, "pred_err_forward"
            ),
            "C1_cross_radius_pred_err_backward_by_speed": summarize(
                curvature_records, "pred_err_backward"
            ),
            "C1_pred_err_forward_norm_by_speed": summarize(
                norm_pred_err(curvature_records), "pred_err_forward_norm"
            ),
            "C2_gradient_cosine_all_by_speed": summarize(valid_grad_records, "cosine"),
            "C2_gradient_magnitude_ratio_all_by_speed": summarize(
                valid_grad_records, "magnitude_ratio"
            ),
            "C2_topk_sign_agreement_all_by_speed": summarize(
                valid_grad_records, "topk_agreement"
            ),
            "C3_cosine_unclipped_by_speed": summarize(
                [r for r in unclipped_records if r["valid_grad"]], "cosine"
            ),
            "C3_cosine_clipped_by_speed": summarize(
                [r for r in clipped_records if r["valid_grad"]], "cosine"
            ),
            "C3_pred_err_forward_clipped_by_speed": summarize(
                clipped_records, "pred_err_forward"
            ),
            "C3_pred_err_forward_unclipped_by_speed": summarize(
                unclipped_records, "pred_err_forward"
            ),
            "A1_any_negative_second_diff_fraction_by_speed": summarize_fraction(
                alpha_records, "any_negative_second_diff"
            ),
            "A3_monotone_descent_fraction_by_speed": summarize_fraction(
                alpha_records, "monotone_descent"
            ),
            "A4_max_negative_curvature_by_speed": summarize(
                alpha_records, "max_negative_curvature"
            ),
        },
        "A2_argmin_location_overall": _argmin_location(alpha_records),
        "caveats": [
            "Cross-radius prediction error is a genuine extrapolation metric, but "
            "still along the 16 Hadamard axes only; off-axis behavior is untested.",
            "Gradient cosine/magnitude use all 16 directional slopes; clipped pairs "
            "bias those slopes, so prefer the unclipped stratification.",
            "Algebraic reconstruction only -- DBM dynamics/cost were not re-run.",
            "Train-only; says nothing about generalization.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"curvature frames: {frame_count}  clipped={len(clipped_records)} "
          f"unclipped={len(unclipped_records)}  alpha lines={len(alpha_records)}")
    for name, block in result["diagnostics"].items():
        print(f"\n{name}")
        for speed, stats in block.items():
            if stats.get("n", 0) == 0:
                continue
            if "mean" in stats:
                median = stats.get("median")
                p95 = stats.get("p95")
                extra = ""
                if median is not None:
                    extra += f" median={median:.5f}"
                if p95 is not None:
                    extra += f" p95={p95:.5f}"
                print(f"  {speed:>8}: n={stats['n']:<6} mean={stats['mean']:.5f}{extra}")
            else:
                print(f"  {speed:>8}: n={stats['n']:<6} fraction={stats['fraction']:.4f}")
    print("\nA2_argmin_location_overall:", result["A2_argmin_location_overall"])
    print(f"\nwrote {args.output}")


def _argmin_location(alpha_records):
    counts = defaultdict(int)
    for record in alpha_records:
        counts[record["argmin_location"]] += 1
    total = len(alpha_records)
    return {key: counts[key] / total for key in ("alpha0", "interior", "alpha1")}


if __name__ == "__main__":
    main()
