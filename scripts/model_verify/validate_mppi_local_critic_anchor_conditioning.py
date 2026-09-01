#!/usr/bin/env python3
"""Independently validate the local-Critic anchor-conditioning artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/direct_local_critic_anchor_conditioning_20260813_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64).reshape(len(left), -1)
    right = np.asarray(right, np.float64).reshape(len(right), -1)
    return np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )


def close(left: float, right: float, tolerance: float = 2e-6) -> None:
    if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"metric mismatch: {left} != {right}")


def main() -> None:
    args = parse_args()
    analysis_path = args.artifact / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["qualification"] not in {
        "ACTION_LOCATION_CONDITIONING_FAIL",
        "ACTION_LOCATION_CONDITIONING_PASS",
    }:
        raise AssertionError("unexpected qualification")
    for name in (
        "critic_summary", "fresh_summary", "fresh_validation", "fresh_fd_audit",
    ):
        path = Path(analysis["sources"][name])
        if sha256_file(path) != analysis["sources"][f"{name}_sha256"]:
            raise AssertionError(f"source hash changed: {name}")
    for path_text, expected in zip(
        analysis["sources"]["critic_checkpoints"],
        analysis["sources"]["critic_checkpoint_sha256"],
    ):
        if sha256_file(Path(path_text)) != expected:
            raise AssertionError("checkpoint hash changed")
    npz_path = Path(analysis["artifacts"]["audit_npz"])
    if sha256_file(npz_path) != analysis["artifacts"]["audit_npz_sha256"]:
        raise AssertionError("audit NPZ hash changed")
    with np.load(npz_path, allow_pickle=False) as archive:
        saved = {key: np.asarray(archive[key]) for key in archive.files}

    left, right = saved["pair_left"], saved["pair_right"]
    if len(left) != 300 or len(right) != 300:
        raise AssertionError("pair count changed")
    if np.any(left + 1 != right):
        raise AssertionError("repeat pairing changed")
    target = saved["target_gradient"]
    true_cosine = cosine_rows(target[left], target[right])
    true_reversal = true_cosine < 0.0
    if not np.array_equal(true_reversal, saved["true_reversal"]):
        raise AssertionError("true reversal mask changed")
    if int(np.sum(true_reversal)) != analysis["counts"][
        "true_gradient_reversal_pair"
    ]:
        raise AssertionError("true reversal count changed")

    step = (
        (saved["actor_center"][right] - saved["actor_center"][left])
        / (2.0 * saved["sigma"][left, None])
    )
    left_directional = np.sum(
        target[left].reshape(-1, 8, 2) * step, axis=(1, 2)
    )
    right_directional = np.sum(
        target[right].reshape(-1, 8, 2) * step, axis=(1, 2)
    )
    if np.max(np.abs(left_directional - saved["left_directional_derivative"])) > 2e-5:
        raise AssertionError("left directional derivative changed")
    if np.max(np.abs(right_directional - saved["right_directional_derivative"])) > 2e-5:
        raise AssertionError("right directional derivative changed")
    positive_to_negative = float(np.mean(
        (left_directional[true_reversal] > 0.0)
        & (right_directional[true_reversal] < 0.0)
    ))
    close(
        positive_to_negative,
        analysis["true_action_response"][
            "reversal_directional_derivative_positive_to_negative_fraction"
        ],
    )

    response_left = cosine_rows(saved["prediction_00"], saved["prediction_01"])
    response_right = cosine_rows(saved["prediction_10"], saved["prediction_11"])
    if np.max(np.abs(response_left - saved["predicted_response_left"])) > 2e-6:
        raise AssertionError("left predicted response changed")
    if np.max(np.abs(response_right - saved["predicted_response_right"])) > 2e-6:
        raise AssertionError("right predicted response changed")
    recall_left = float(np.mean(response_left[true_reversal] < 0.0))
    recall_right = float(np.mean(response_right[true_reversal] < 0.0))
    close(
        recall_left,
        analysis["critic_action_response"]["true_reversal_flip_recall_left"],
    )
    close(
        recall_right,
        analysis["critic_action_response"]["true_reversal_flip_recall_right"],
    )
    # v1 artifacts predate the explicit own-action fresh-target arrays.  New
    # artifacts must carry and independently revalidate them.
    if "own_prediction" in saved:
        own_prediction = saved["own_prediction"]
        own_cosine = cosine_rows(own_prediction, target)
        own_norm_ratio = np.linalg.norm(own_prediction, axis=1) / (
            np.linalg.norm(target, axis=1) + 1e-12
        )
        close(
            float(np.median(own_cosine)),
            analysis["critic_action_response"]["own_fresh_target_cosine"]["median"],
        )
        close(
            float(np.median(own_norm_ratio)),
            analysis["critic_action_response"]["own_fresh_target_norm_ratio"]["median"],
        )

    validation = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "physical_pair_count": int(len(left)),
        "true_reversal_count": int(np.sum(true_reversal)),
        "positive_to_negative_fraction": positive_to_negative,
        "critic_flip_recall_left": recall_left,
        "critic_flip_recall_right": recall_right,
        "critic_response_cosine_median_left": float(np.median(response_left)),
        "critic_response_cosine_median_right": float(np.median(response_right)),
        "contract": analysis["contract"],
    }
    path = args.artifact / "validation_summary.json"
    path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
