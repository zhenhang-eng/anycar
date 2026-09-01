#!/usr/bin/env python3
"""Independently validate the zero-rollout local-Critic hard-state artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v2"
)
SPACES = (
    "raw_equal_block", "actor_encoder", "critic_ensemble_representation",
)
EXPECTED_THRESHOLDS = {
    "hard_critic_cosine_lt": 0.0,
    "label_mismatch_cosine_lt": 0.90,
    "label_warning_cosine_lt": 0.95,
    "low_density_percentile_lte": 0.10,
    "mixed_neighbor_coherence_lt": 0.30,
    "wrong_coherent_mean_label_cosine_lt": -0.50,
    "wrong_coherent_neighbor_coherence_gt": 0.50,
    "within_state_repeat_cosine_lt": 0.50,
    "adjacent_physical_state_cosine_lt": 0.50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def close(left: float, right: float, tolerance: float = 2e-6) -> None:
    if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"metric mismatch: {left} != {right}")


def main() -> None:
    args = parse_args()
    analysis_path = args.artifact / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["format_version"] != 2:
        raise AssertionError("expected v2 hard-state analysis")
    if analysis["qualification"] != "HARD_STATE_ZERO_ROLLOUT_ATTRIBUTION_COMPLETE":
        raise AssertionError("unexpected hard-state qualification")
    expected_contract = {
        "new_dbm_rollouts": 0,
        "actor_updated": False,
        "critic_updated": False,
        "formal_validation_loaded": False,
        "test_loaded": False,
        "explicit_gradient_head_already_present": True,
    }
    if analysis["contract"] != expected_contract:
        raise AssertionError("hard-state contract changed")
    for key in (
        "critic_summary", "fresh_summary", "fresh_validation", "previous_analysis",
        "local_forward_labels", "fresh_fd_audit",
    ):
        path = Path(analysis["sources"][key])
        if sha256_file(path) != analysis["sources"][f"{key}_sha256"]:
            raise AssertionError(f"source hash mismatch: {path}")
    if json.loads(Path(analysis["sources"]["fresh_validation"]).read_text())[
        "qualification"
    ] != "PASS":
        raise AssertionError("fresh source validation is no longer PASS")

    manifest_path = Path(analysis["artifacts"]["hard_state_manifest"])
    context_manifest_path = Path(
        analysis["artifacts"]["context_attribution_manifest"]
    )
    npz_path = Path(analysis["artifacts"]["analysis_npz"])
    if sha256_file(context_manifest_path) != analysis["artifacts"][
        "context_attribution_manifest_sha256"
    ]:
        raise AssertionError("context manifest hash mismatch")
    if sha256_file(manifest_path) != analysis["artifacts"][
        "hard_state_manifest_sha256"
    ]:
        raise AssertionError("hard-state manifest hash mismatch")
    if sha256_file(npz_path) != analysis["artifacts"]["analysis_npz_sha256"]:
        raise AssertionError("hard-state NPZ hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    context_manifest = json.loads(context_manifest_path.read_text())
    for local in (manifest, context_manifest):
        if local["format_version"] != 2:
            raise AssertionError("manifest format changed")
        if local["thresholds_pre_registered"] != EXPECTED_THRESHOLDS:
            raise AssertionError("pre-registered attribution thresholds changed")

    with np.load(npz_path, allow_pickle=False) as archive:
        saved = {key: np.asarray(archive[key]) for key in archive.files}
    target = saved["target"]
    critic_gradient = saved["critic_gradient"]
    ensemble = np.mean(critic_gradient, axis=0)
    cosine = cosine_rows(ensemble, target)
    hard = cosine < 0.0
    if not np.array_equal(ensemble, saved["ensemble"]):
        raise AssertionError("saved ensemble is not exact critic mean")
    if np.max(np.abs(cosine - saved["ensemble_cosine"])) > 2e-6:
        raise AssertionError("saved ensemble cosine changed")
    if not np.array_equal(hard, saved["hard"]):
        raise AssertionError("saved hard mask changed")
    if int(np.sum(hard)) != analysis["counts"]["hard_context"]:
        raise AssertionError("hard-state count changed")
    if manifest["hard_count"] != int(np.sum(hard)) or len(manifest["rows"]) != int(
        np.sum(hard)
    ):
        raise AssertionError("manifest hard count changed")
    expected_context = saved["heldout_context"][hard]
    manifest_context = np.asarray(
        [row["context_index"] for row in manifest["rows"]], np.int64
    )
    if not np.array_equal(expected_context, manifest_context):
        raise AssertionError("manifest context order changed")
    if context_manifest["context_count"] != len(hard):
        raise AssertionError("context manifest count changed")
    if len(context_manifest["rows"]) != len(hard):
        raise AssertionError("context manifest row count changed")
    if context_manifest["hard_count"] != int(np.sum(hard)):
        raise AssertionError("context manifest hard count changed")
    all_manifest_context = np.asarray(
        [row["context_index"] for row in context_manifest["rows"]], np.int64
    )
    if not np.array_equal(all_manifest_context, saved["heldout_context"]):
        raise AssertionError("context manifest ordering changed")

    label_cosine = cosine_rows(saved["stored_smallest_gradient"], target)
    label_norm_ratio = np.linalg.norm(
        saved["stored_smallest_gradient"], axis=1
    ) / (np.linalg.norm(target, axis=1) + 1e-12)
    if np.max(np.abs(label_cosine - saved["label_fresh_cosine"])) > 2e-6:
        raise AssertionError("stored label/fresh cosine changed")
    if np.max(np.abs(label_norm_ratio - saved["label_fresh_norm_ratio"])) > 2e-6:
        raise AssertionError("stored label/fresh norm ratio changed")
    close(
        float(np.quantile(label_cosine[hard], 0.10)),
        analysis["label_self_check"]["cosine"]["hard"]["p10"],
    )
    if int(np.sum(label_cosine < 0.0)) != 0:
        raise AssertionError("stored 0.05 label unexpectedly reverses fresh FD")

    physical = saved["physical_snapshot_ordinal"]
    repeat = saved["repeat_index"]
    ordinal = saved["context_ordinal"]
    for episode in np.unique(saved["heldout_episode"]):
        local = np.flatnonzero(saved["heldout_episode"] == episode)
        if len(local) != 10:
            raise AssertionError("unexpected per-episode context count")
        if not np.array_equal(ordinal[local], np.arange(10)):
            raise AssertionError("episode context ordinal changed")
        if not np.array_equal(physical[local], np.repeat(np.arange(5), 2)):
            raise AssertionError("physical snapshot ordinal changed")
        if not np.array_equal(repeat[local], np.tile((0, 1), 5)):
            raise AssertionError("repeat index changed")

    route_keys = sorted(
        key[len("route_"):] for key in saved if key.startswith("route_")
    )
    if sorted(analysis["routing"]["flag_summary"]) != route_keys:
        raise AssertionError("route flag summary keys changed")
    for row_index, row in enumerate(context_manifest["rows"]):
        expected_flags = {
            name: bool(saved[f"route_{name}"][row_index]) for name in route_keys
        }
        if row["route_flags"] != expected_flags:
            raise AssertionError(f"route flags changed at row {row_index}")
        if row["route_tags"] != [
            name for name, value in expected_flags.items() if value
        ]:
            raise AssertionError(f"route tags changed at row {row_index}")
        if row["physical_snapshot_ordinal"] != int(physical[row_index]):
            raise AssertionError("manifest physical ordinal changed")
        if row["repeat_index"] != int(repeat[row_index]):
            raise AssertionError("manifest repeat index changed")

    reconstructed_routes = {
        "label_mismatch": label_cosine
        < EXPECTED_THRESHOLDS["label_mismatch_cosine_lt"],
        "label_warning": (
            (label_cosine < EXPECTED_THRESHOLDS["label_warning_cosine_lt"])
            & (label_cosine >= EXPECTED_THRESHOLDS["label_mismatch_cosine_lt"])
        ),
        "boundary_clipped": saved["clipped"].astype(bool),
        "cold_start_physical_snapshot": physical == 0,
        "within_state_repeat_transition": np.isfinite(saved["repeat_cosine"])
        & (saved["repeat_cosine"]
           < EXPECTED_THRESHOLDS["within_state_repeat_cosine_lt"]),
        "adjacent_physical_state_transition": np.isfinite(saved["temporal_cosine"])
        & (saved["temporal_cosine"]
           < EXPECTED_THRESHOLDS["adjacent_physical_state_cosine_lt"]),
    }
    for space in SPACES:
        reconstructed_routes[f"low_density_{space}"] = (
            saved[f"{space}_density_percentile"]
            <= EXPECTED_THRESHOLDS["low_density_percentile_lte"]
        )
        reconstructed_routes[f"mixed_neighbors_{space}"] = (
            saved[f"{space}_coherence"]
            < EXPECTED_THRESHOLDS["mixed_neighbor_coherence_lt"]
        )
    reconstructed_routes["wrong_coherent_critic_branch"] = (
        (saved["critic_ensemble_representation_mean_label_cosine"]
         < EXPECTED_THRESHOLDS["wrong_coherent_mean_label_cosine_lt"])
        & (saved["critic_ensemble_representation_coherence"]
           > EXPECTED_THRESHOLDS[
               "wrong_coherent_neighbor_coherence_gt"
           ])
    )
    if sorted(reconstructed_routes) != route_keys:
        raise AssertionError("independent route keys changed")
    for name, value in reconstructed_routes.items():
        if not np.array_equal(value, saved[f"route_{name}"]):
            raise AssertionError(f"route reconstruction mismatch: {name}")
        summary = analysis["routing"]["flag_summary"][name]
        close(float(np.mean(value[hard])), summary["hard_coverage_fraction"])
        close(float(np.mean(hard[value])), summary["hard_rate_with_flag"])
        close(float(np.mean(hard[~value])), summary["hard_rate_without_flag"])

    train_gradient = normalize_rows(saved["train_gradient"])
    normalized_target = normalize_rows(target)
    replay = {}
    for space in SPACES:
        index = saved[f"{space}_top_index"]
        similarity = saved[f"{space}_top_similarity"]
        if index.shape != (len(target), 20) or similarity.shape != index.shape:
            raise AssertionError(f"unexpected top-k shape: {space}")
        if np.any(index < 0) or np.any(index >= len(train_gradient)):
            raise AssertionError(f"neighbor index out of range: {space}")
        if np.any(np.diff(similarity, axis=1) > 1e-7):
            raise AssertionError(f"neighbor similarity is not sorted: {space}")
        if any(len(np.unique(row)) != len(row) for row in index):
            raise AssertionError(f"duplicate top-k neighbor: {space}")
        neighbor = train_gradient[index]
        mean_direction = np.mean(neighbor, axis=1)
        mean_cosine = cosine_rows(mean_direction, normalized_target)
        coherence = np.linalg.norm(mean_direction, axis=1)
        individual = np.sum(neighbor * normalized_target[:, None], axis=2)
        stored = analysis["representation_spaces"][space]
        close(
            float(np.median(similarity[hard, 0])),
            stored["top1_similarity_hard_vs_nonhard"]["hard"]["median"],
        )
        close(
            float(np.median(similarity[~hard, 0])),
            stored["top1_similarity_hard_vs_nonhard"]["nonhard"]["median"],
        )
        close(
            float(np.median(mean_cosine[hard])),
            stored["neighbors"]["20"]["hard_mean_direction_cosine"]["median"],
        )
        close(
            float(np.median(coherence[hard])),
            stored["neighbors"]["20"]["hard_neighbor_direction_coherence"]["median"],
        )
        close(
            float(np.median(np.max(individual[hard], axis=1))),
            stored["neighbors"]["20"]["hard_best_neighbor_cosine"]["median"],
        )
        replay[space] = {
            "hard_top1_similarity_median": float(np.median(similarity[hard, 0])),
            "hard_top20_mean_direction_cosine_median": float(
                np.median(mean_cosine[hard])
            ),
            "hard_top20_coherence_median": float(np.median(coherence[hard])),
            "hard_top20_best_neighbor_cosine_median": float(
                np.median(np.max(individual[hard], axis=1))
            ),
        }

    normalized_prediction = normalize_rows(ensemble)
    alignment = normalized_prediction * normalized_target
    acceleration = np.arange(0, 16, 2)
    steering = np.arange(1, 16, 2)
    negative_total = np.sum(np.maximum(-alignment[hard], 0.0))
    steering_fraction = float(
        np.sum(np.maximum(-alignment[hard][:, steering], 0.0)) / negative_total
    )
    acceleration_fraction = float(
        np.sum(np.maximum(-alignment[hard][:, acceleration], 0.0)) / negative_total
    )
    close(
        steering_fraction,
        analysis["component_decomposition"]["by_action_channel"]["steering"][
            "negative_alignment_mass_fraction"
        ],
    )

    target_norm = np.linalg.norm(target, axis=1)
    parallel_scale = np.sum(ensemble * target, axis=1) / (
        np.square(target_norm) + 1e-12
    )
    parallel = parallel_scale[:, None] * target
    perpendicular_ratio = np.linalg.norm(ensemble - parallel, axis=1) / (
        target_norm + 1e-12
    )
    if np.max(np.abs(parallel_scale - saved["parallel_scale"])) > 2e-6:
        raise AssertionError("parallel decomposition changed")
    if np.max(np.abs(perpendicular_ratio - saved["perpendicular_ratio"])) > 2e-6:
        raise AssertionError("perpendicular decomposition changed")
    close(
        float(np.median(parallel_scale[hard])),
        analysis["component_decomposition"]["parallel_perpendicular"][
            "parallel_scale_hard_vs_nonhard"
        ]["hard"]["median"],
    )
    close(
        acceleration_fraction,
        analysis["component_decomposition"]["by_action_channel"]["acceleration"][
            "negative_alignment_mass_fraction"
        ],
    )

    validation = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "hard_context_count": int(np.sum(hard)),
        "manifest_row_count": len(manifest["rows"]),
        "context_manifest_row_count": len(context_manifest["rows"]),
        "label_self_check": {
            "hard_cosine_p10": float(np.quantile(label_cosine[hard], 0.10)),
            "negative_count": int(np.sum(label_cosine < 0.0)),
            "hard_label_mismatch_count": int(np.sum(
                hard & (label_cosine < EXPECTED_THRESHOLDS[
                    "label_mismatch_cosine_lt"
                ])
            )),
        },
        "route_flag_count": len(route_keys),
        "spaces": replay,
        "steering_negative_alignment_mass_fraction": steering_fraction,
        "hard_parallel_scale_median": float(np.median(parallel_scale[hard])),
        "hard_perpendicular_ratio_median": float(
            np.median(perpendicular_ratio[hard])
        ),
        "acceleration_negative_alignment_mass_fraction": acceleration_fraction,
        "contract": {
            "new_dbm_rollouts": 0,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "actor_updated": False,
            "critic_updated": False,
        },
    }
    path = args.artifact / "validation_summary.json"
    path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({
        "qualification": validation["qualification"],
        "hard_context_count": validation["hard_context_count"],
        "label_self_check": validation["label_self_check"],
        "route_flag_count": validation["route_flag_count"],
        "spaces": validation["spaces"],
        "steering_negative_alignment_mass_fraction": steering_fraction,
    }, indent=2))


if __name__ == "__main__":
    main()
