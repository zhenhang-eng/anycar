#!/usr/bin/env python3
"""Output-side temporal coupling diagnostic (zero rollout, residual basis).

Tests whether the actor's predicted knots reproduce the teacher's
cross-knot temporal structure on the residual relative to each state's
anchor a0: per-knot residual variance distribution (leverage expression),
cross-knot residual covariance reproduction, and residual trajectory
smoothness. Compares base (one-shot flat head) against A2-G/T/GT (token
heads) using the stored n1800 OOF predictions and the consensus-64 teacher
labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)


DEFAULT_POOL_LABELS = Path(
    "outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/labels.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/output_temporal_coupling_diagnostic_20260818_v1"
)
ARCHS = {
    "base": Path(
        "outputs/mppi_proposal/actor_curve_n1800_predump_20260818_v1/"
        "oof_evaluation.npz"
    ),
    "A2-G": Path(
        "outputs/mppi_proposal/a2_g_n1800_20260818_v1/oof_evaluation.npz"
    ),
    "A2-T": Path(
        "outputs/mppi_proposal/a2_t_n1800_20260818_v1/oof_evaluation.npz"
    ),
    "A2-GT": Path(
        "outputs/mppi_proposal/a2_gt_n1800_20260818_v1/oof_evaluation.npz"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-labels", type=Path, default=DEFAULT_POOL_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def upper_offdiag(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    labels = dict(np.load(args.pool_labels, allow_pickle=False))
    teacher_knots = labels["label_knots"].astype(np.float32)
    if teacher_knots.shape != (1800, 8, 2):
        raise AssertionError(f"unexpected teacher shape {teacher_knots.shape}")
    teacher_keys = [str(value) for value in labels["episodes"]]

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=0,
    )
    states = select_states(load_states(loader_args), 1800)
    a0_by_key = {
        f"{state['episode']}#{state['snapshot']}": state["a0"].astype(np.float32)
        for state in states
    }
    missing = [key for key in teacher_keys if key not in a0_by_key]
    if missing:
        raise AssertionError(f"{len(missing)} teacher keys lack a0")
    a0 = np.stack([a0_by_key[key] for key in teacher_keys])
    teacher_res = teacher_knots - a0

    results = {}
    for arch, path in ARCHS.items():
        data = dict(np.load(path, allow_pickle=True))
        predicted = data["predicted_knots"].astype(np.float32)
        seeds = data["seed"].astype(np.int64)
        keys = [str(value) for value in data["state_keys"]]
        if predicted.shape != (5400, 8, 2):
            raise AssertionError(
                f"{arch}: unexpected predicted shape {predicted.shape}"
            )
        per_seed = {}
        for seed in np.unique(seeds):
            seed_rows = np.flatnonzero(seeds == seed)
            pred = predicted[seed_rows]
            if len(pred) != 5400 // 3:
                raise AssertionError(
                    f"{arch} seed{seed}: {len(pred)} rows"
                )
            seed_key_pos = {
                keys[row]: local
                for local, row in enumerate(seed_rows)
            }
            pred = pred[[seed_key_pos[key] for key in teacher_keys]]
            pred_res = pred - a0
            t_var = teacher_res.var(axis=0)
            p_var = pred_res.var(axis=0)
            t_dist = t_var.sum(axis=1) / (t_var.sum() + 1e-12)
            p_dist = p_var.sum(axis=1) / (p_var.sum() + 1e-12)
            cov_entries = []
            for channel in range(2):
                cov_t = np.cov(teacher_res[:, :, channel].T)
                cov_p = np.cov(pred_res[:, :, channel].T)
                vt = upper_offdiag(cov_t)
                vp = upper_offdiag(cov_p)
                cov_entries.append({
                    "channel": channel,
                    "cov_vector_correlation": float(np.corrcoef(vt, vp)[0, 1]),
                    "cov_scale_ratio": float(
                        (np.std(vp) + 1e-12) / (np.std(vt) + 1e-12)
                    ),
                })
            smooth_t = float(np.mean(np.square(np.diff(teacher_res, axis=1))))
            smooth_p = float(np.mean(np.square(np.diff(pred_res, axis=1))))
            per_knot_corr = [
                float(np.corrcoef(
                    pred_res[:, k, c], teacher_res[:, k, c]
                )[0, 1])
                for k in range(8) for c in range(2)
            ]
            per_seed[str(int(seed))] = {
                "variance_distribution_teacher": t_dist.tolist(),
                "variance_distribution_predicted": p_dist.tolist(),
                "variance_dist_pearson": float(np.corrcoef(t_dist, p_dist)[0, 1]),
                "cross_knot_covariance": cov_entries,
                "smoothness_teacher": smooth_t,
                "smoothness_predicted": smooth_p,
                "smoothness_ratio": float(smooth_p / (smooth_t + 1e-12)),
                "per_knot_residual_corr_median": float(np.median(per_knot_corr)),
                "per_knot_residual_corr_early02_median": float(np.median(
                    per_knot_corr[:6]
                )),
            }
        results[arch] = per_seed

    summary_rows = {}
    for arch, per_seed in results.items():
        cov_corrs = [
            entry["cov_vector_correlation"]
            for seed_data in per_seed.values()
            for entry in seed_data["cross_knot_covariance"]
        ]
        cov_scales = [
            entry["cov_scale_ratio"]
            for seed_data in per_seed.values()
            for entry in seed_data["cross_knot_covariance"]
        ]
        var_pearsons = [
            seed_data["variance_dist_pearson"] for seed_data in per_seed.values()
        ]
        smooth_ratios = [
            seed_data["smoothness_ratio"] for seed_data in per_seed.values()
        ]
        knot_corrs = [
            seed_data["per_knot_residual_corr_median"]
            for seed_data in per_seed.values()
        ]
        summary_rows[arch] = {
            "cross_knot_cov_vector_corr_median": float(np.median(cov_corrs)),
            "cross_knot_cov_scale_ratio_median": float(np.median(cov_scales)),
            "variance_dist_pearson_median": float(np.median(var_pearsons)),
            "smoothness_ratio_median": float(np.median(smooth_ratios)),
            "per_knot_residual_corr_median": float(np.median(knot_corrs)),
            "per_seed": per_seed,
        }
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OUTPUT_TEMPORAL_COUPLING_DIAGNOSED_ACTOR_FROZEN",
        "sources": {
            "pool_labels": str(args.pool_labels.resolve()),
            "architectures": {k: str(v) for k, v in ARCHS.items()},
        },
        "metrics": {
            "basis": "residuals relative to per-state anchor a0",
            "cross_knot_cov_vector_corr": (
                "Pearson between teacher and predicted upper-off-diagonal "
                "cross-knot residual covariance vectors, per channel"
            ),
            "variance_dist_pearson": (
                "Pearson between teacher/predicted per-knot residual "
                "variance shares"
            ),
            "smoothness_ratio": (
                "mean squared adjacent-knot residual difference, "
                "predicted / teacher"
            ),
        },
        "results": summary_rows,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "zero_rollout": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"{'arch':7s} {'covvec_corr':>12s} {'cov_scale':>10s} "
          f"{'var_dist_r':>11s} {'smooth_r':>9s} {'knot_corr':>10s}")
    for arch, row in summary_rows.items():
        print(f"{arch:7s} {row['cross_knot_cov_vector_corr_median']:12.3f} "
              f"{row['cross_knot_cov_scale_ratio_median']:10.3f} "
              f"{row['variance_dist_pearson_median']:11.3f} "
              f"{row['smoothness_ratio_median']:9.3f} "
              f"{row['per_knot_residual_corr_median']:10.3f}")
    t = results["base"]["0"]["variance_distribution_teacher"]
    print("\nresidual variance distribution over knots (seed 0):")
    for arch in ARCHS:
        p = results[arch]["0"]["variance_distribution_predicted"]
        print(f"  {arch:7s} pred: {[round(v,3) for v in p]}")
    print(f"  teacher : {[round(v,3) for v in t]}")


if __name__ == "__main__":
    main()
