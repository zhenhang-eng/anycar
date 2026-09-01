#!/usr/bin/env python3
"""Paired old-PA vs new-PA recomputation on the untouched 230-frame subset.

The 230 frames are the complete-episode fresh-FD subset whose episodes were
never consumed by targeted response training. Both runs are evaluated on
exactly the same rows, per seed, so the delta is a paired comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPISemanticStructuredLocalQCritic,
)
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_structured_local_q_critic import (
    fresh_metrics,
    load_npz,
    subset_first_axis,
)


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_TARGETED = Path(
    "outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_TARGETED_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--targeted-summary", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--targeted-labels", type=Path, default=DEFAULT_TARGETED_LABELS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_summary.read_text())
    targeted_summary = json.loads(args.targeted_summary.read_text())
    if baseline["sources"]["initial_actor_sha256"] != targeted_summary["sources"]["initial_actor_sha256"]:
        raise AssertionError("pre/post runs use different frozen Actors")
    if baseline["sources"]["labels_sha256"] != targeted_summary["sources"]["labels_sha256"]:
        raise AssertionError("pre/post runs use different base labels")
    if baseline["sources"]["fresh_fd_sha256"] != targeted_summary["sources"]["fresh_fd_sha256"]:
        raise AssertionError("pre/post runs use different fresh-FD audits")
    if sha256_file(args.targeted_labels) != targeted_summary["sources"]["targeted_labels_sha256"]:
        raise AssertionError("targeted label hash mismatch")

    initial_path = Path(baseline["sources"]["initial_actor"])
    initial = torch.load(initial_path, map_location="cpu")
    alpha_payload = torch.load(initial["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy,
        tensors,
        extra,
        np.arange(len(data.episodes)),
        float(initial["base_move_threshold"]),
        args.batch_size,
        device,
    )
    inputs = list(actor_inputs(tensors, alpha_center, device))
    base_labels = load_npz(Path(baseline["sources"]["labels"]))
    action_by_context = np.zeros((len(data.episodes), 8, 2), np.float32)
    assigned = np.zeros(len(data.episodes), bool)
    for position, context in enumerate(base_labels["context_index"].astype(np.int64)):
        action_by_context[context] = base_labels["actor_center"][position]
        assigned[context] = True
    if not np.all(assigned):
        raise AssertionError("absolute action map is incomplete")
    inputs[3] = torch.from_numpy(action_by_context).to(device)
    inputs = tuple(inputs)

    fresh = load_npz(Path(baseline["sources"]["fresh_fd"]))
    targeted_labels = load_npz(args.targeted_labels)
    fresh_context = fresh["context_index"].astype(np.int64)
    consumed_context = set(
        targeted_labels["context_index"][
            targeted_labels["pilot_role"] == "target"
        ].astype(np.int64).tolist()
    )
    consumed_episode = set(
        fresh["episode"][np.asarray(
            [int(value) in consumed_context for value in fresh_context], bool
        )].tolist()
    )
    unseen_positions = np.asarray(
        [
            position for position, value in enumerate(fresh["episode"])
            if value not in consumed_episode
        ],
        np.int64,
    )
    if len(unseen_positions) != 230 or len(unseen_positions) % 2:
        raise AssertionError(
            f"untouched complete-episode subset mismatch: {len(unseen_positions)}"
        )
    fresh_unseen = subset_first_axis(fresh, unseen_positions)

    eval_args = SimpleNamespace(
        evaluation_batch_size=args.batch_size,
        small_chord_sigma=0.15,
        medium_chord_sigma=0.30,
    )
    maximum_residual_sigma = float(initial["maximum_residual_sigma"])

    summaries = {"baseline": baseline, "targeted": targeted_summary}
    records: dict[str, dict[str, dict]] = {}
    checkpoint_hash_checks = {}
    for run_name, summary in summaries.items():
        run_records = {
            int(record["seed"]): record
            for record in summary["arms"]["PA"]["records"]
        }
        records[run_name] = {}
        for seed in (0, 1, 2):
            record = run_records[seed]
            checkpoint = Path(record["checkpoint"])
            checkpoint_hash_checks[f"{run_name}_seed{seed}"] = (
                sha256_file(checkpoint) == record["checkpoint_sha256"]
            )
            payload = torch.load(checkpoint, map_location="cpu")
            model = TorchMPPISemanticStructuredLocalQCritic(
                include_feedback=bool(payload["include_feedback"]),
                low_rank=int(payload["low_rank"]),
                dropout=0.0,
                hessian_scale=float(payload["hessian_scale"]),
                hessian_enabled=bool(payload["hessian_enabled"]),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            records[run_name][str(seed)] = fresh_metrics(
                model, inputs, fresh_unseen, maximum_residual_sigma,
                eval_args, device,
            )
    if not all(checkpoint_hash_checks.values()):
        raise AssertionError("checkpoint hash mismatch")

    metric_names = (
        "gradient_cosine_median",
        "gradient_cosine_p10",
        "gradient_positive_fraction",
        "gradient_norm_ratio_median",
        "true_reversal_flip_recall",
    )
    paired_delta = {}
    for metric in metric_names:
        values = [
            records["targeted"][str(seed)][metric]
            - records["baseline"][str(seed)][metric]
            for seed in (0, 1, 2)
        ]
        paired_delta[metric] = {"by_seed": values, **distribution(values)}
    small_chord_delta = {}
    for metric in ("pair_count", "reversal_pair_count", "reversal_flip_recall"):
        values = [
            (records["targeted"][str(seed)]["chord_bins"]["small_le_0_15"][metric] or 0.0)
            - (records["baseline"][str(seed)]["chord_bins"]["small_le_0_15"][metric] or 0.0)
            for seed in (0, 1, 2)
        ]
        small_chord_delta[metric] = {"by_seed": values, **distribution(values)}
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "UNTOUCHED_230_PAIRED_COMPARISON_ACTOR_FROZEN",
        "sources": {
            "baseline_summary": str(args.baseline_summary.resolve()),
            "baseline_summary_sha256": sha256_file(args.baseline_summary),
            "targeted_summary": str(args.targeted_summary.resolve()),
            "targeted_summary_sha256": sha256_file(args.targeted_summary),
            "targeted_labels": str(args.targeted_labels.resolve()),
            "targeted_labels_sha256": sha256_file(args.targeted_labels),
            "fresh_fd": str(Path(baseline["sources"]["fresh_fd"]).resolve()),
            "fresh_fd_sha256": baseline["sources"]["fresh_fd_sha256"],
        },
        "subset": {
            "definition": (
                "fresh-FD rows whose episode never appears in targeted "
                "target-context training (complete-episode removal)"
            ),
            "frame_count": int(len(unseen_positions)),
            "episode_count": int(len(np.unique(fresh_unseen["episode"]))),
            "note": (
                "This is the untouched complete-episode subset, not the "
                "original 600-frame distribution."
            ),
        },
        "checkpoint_hash_checks": checkpoint_hash_checks,
        "records": records,
        "paired_targeted_minus_baseline": paired_delta,
        "paired_small_chord_targeted_minus_baseline": small_chord_delta,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "paired_same_rows_same_seed": True,
        },
    }
    output = args.output or args.targeted_summary.with_name(
        "unseen230_paired_comparison.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "qualification": result["qualification"],
        "subset": result["subset"],
        "paired_targeted_minus_baseline": paired_delta,
        "paired_small_chord_targeted_minus_baseline": small_chord_delta,
    }, indent=2))


if __name__ == "__main__":
    main()
