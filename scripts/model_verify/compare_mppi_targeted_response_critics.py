#!/usr/bin/env python3
"""Compare pre/post-targeted Critic checkpoints on identical response labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

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
    load_npz,
    targeted_metrics,
)


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_TARGETED = Path(
    "outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--targeted-summary", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
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
    labels = load_npz(args.labels)
    if sha256_file(args.labels) != targeted_summary["sources"]["targeted_labels_sha256"]:
        raise AssertionError("targeted label hash mismatch")
    if baseline["sources"]["initial_actor_sha256"] != targeted_summary["sources"]["initial_actor_sha256"]:
        raise AssertionError("pre/post runs use different frozen Actors")
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

    role_rows = {
        "target_79": np.flatnonzero(labels["pilot_role"] == "target"),
        "matched_easy_control_21": np.flatnonzero(
            labels["pilot_role"] == "matched_easy_control"
        ),
    }
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
            actual_hash = sha256_file(checkpoint)
            checkpoint_hash_checks[f"{run_name}_seed{seed}"] = (
                actual_hash == record["checkpoint_sha256"]
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
            records[run_name][str(seed)] = {
                role: targeted_metrics(
                    model, inputs, labels, rows, args.batch_size, device
                )
                for role, rows in role_rows.items()
            }
    if not all(checkpoint_hash_checks.values()):
        raise AssertionError("checkpoint hash mismatch")

    metric_names = (
        "gradient_cosine_median",
        "gradient_cosine_p10",
        "gradient_norm_ratio_median",
        "reversal_flip_recall",
    )
    paired_delta = {}
    for role in role_rows:
        paired_delta[role] = {}
        for metric in metric_names:
            values = [
                records["targeted"][str(seed)][role][metric]
                - records["baseline"][str(seed)][role][metric]
                for seed in (0, 1, 2)
            ]
            paired_delta[role][metric] = {
                "by_seed": values,
                **distribution(values),
            }
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "TARGETED_RESPONSE_PAIRED_COMPARISON_ACTOR_FROZEN",
        "sources": {
            "baseline_summary": str(args.baseline_summary.resolve()),
            "baseline_summary_sha256": sha256_file(args.baseline_summary),
            "targeted_summary": str(args.targeted_summary.resolve()),
            "targeted_summary_sha256": sha256_file(args.targeted_summary),
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_file(args.labels),
        },
        "checkpoint_hash_checks": checkpoint_hash_checks,
        "records": records,
        "paired_targeted_minus_baseline": paired_delta,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "target_79_seen_by_targeted_run": True,
            "matched_easy_control_21_seen_by_targeted_run": False,
        },
    }
    output = args.output or args.targeted_summary.with_name(
        "targeted_response_comparison.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "qualification": result["qualification"],
        "paired_targeted_minus_baseline": paired_delta,
    }, indent=2))


if __name__ == "__main__":
    main()
