#!/usr/bin/env python3
"""Independent validator for the grouped-CV run: strict pooled OOF metrics
and g0/H decomposition.

Answers two attribution questions the absolute-gradient cosine cannot:
1. Is the actor-center gradient g0 itself transferred (per stratum)?
2. Does the centered local response H(a0)(a-a0) match the true centered
   response g_true(a)-g_true(a0), independent of g0's direction?

Also recomputes the absolute OOF cosine as a strict pooled median/P10 by
concatenating all out-of-fold predictions (per seed, and across all runs),
replacing the location-weighted fold-median mean used in the run summary.
"""

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
from train_mppi_direct_local_gradient_critic import cosine_rows
from train_mppi_structured_local_q_critic import (
    load_npz,
    location_partner,
    parameters_at_absolute_actions,
)
from run_mppi_targeted_grouped_cv import build_state_table


DEFAULT_CV_SUMMARY = Path(
    "outputs/mppi_proposal/targeted_grouped_cv_20260814_v1/summary.json"
)
DEFAULT_TARGETED_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--targeted-labels", type=Path, default=DEFAULT_TARGETED_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def predict_locations(
    model: TorchMPPISemanticStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    contexts: np.ndarray,
    absolute_actions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    gradients, hessians = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            stop = min(start + batch_size, len(contexts))
            _, gradient, hessian = parameters_at_absolute_actions(
                model,
                inputs,
                torch.from_numpy(contexts[start:stop]).to(device),
                torch.from_numpy(absolute_actions[start:stop]).to(device),
            )
            gradients.append(gradient.cpu().numpy())
            hessians.append(hessian.cpu().numpy())
    return (
        np.concatenate(gradients).astype(np.float32),
        np.concatenate(hessians).astype(np.float32),
    )


def response_block(
    gradient_true: np.ndarray,
    gradient_pred: np.ndarray,
    hessian_center: np.ndarray,
    local_actions: np.ndarray,
    partner: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-location arrays for one context (77 locations).

    gradient_true/pred: (77, 16) at each location; hessian_center: (16, 16)
    at the actor center; local_actions: (77, 16) local action coordinates.
    """
    delta = local_actions - local_actions[0]
    centered_true = gradient_true - gradient_true[0]
    centered_h = np.einsum("ij,bj->bi", hessian_center, delta)
    absolute_cosine = cosine_rows(gradient_pred, gradient_true)
    g0_cosine = cosine_rows(
        gradient_pred[0][None], gradient_true[0][None]
    )[0]
    # The center location has delta=0 (degenerate cosine 0); exclude it from
    # every centered / counterfactual metric.
    outer = np.arange(1, len(local_actions))
    counterfactual_h = gradient_true[0][None] + centered_h
    counterfactual_g0 = gradient_pred[0][None] + centered_true
    counterfactual_full = gradient_pred[0][None] + centered_h
    return {
        "absolute_cosine": absolute_cosine,
        "g0_center_cosine": g0_cosine,
        "centered_cosine": cosine_rows(centered_h[outer], centered_true[outer]),
        "centered_norm_ratio": np.linalg.norm(centered_h[outer], axis=1) / (
            np.linalg.norm(centered_true[outer], axis=1) + 1e-12
        ),
        "cf_true_g0_plus_pred_h_cosine": cosine_rows(
            counterfactual_h[outer], gradient_true[outer]
        ),
        "cf_pred_g0_plus_true_delta_cosine": cosine_rows(
            counterfactual_g0[outer], gradient_true[outer]
        ),
        "cf_pred_g0_plus_pred_h_cosine": cosine_rows(
            counterfactual_full[outer], gradient_true[outer]
        ),
        "cf_full_norm_ratio": np.linalg.norm(
            counterfactual_full[outer], axis=1
        ) / (np.linalg.norm(gradient_true[outer], axis=1) + 1e-12),
        "true_pair_reversal": true_pair_mask(gradient_true, partner),
        "h_pair_reversal_pred": h_pair_mask(centered_h, partner),
        "g0_norm_ratio": (
            np.linalg.norm(gradient_pred[0]) / (np.linalg.norm(gradient_true[0]) + 1e-12)
        ),
    }


def true_pair_mask(gradient_true: np.ndarray, partner: np.ndarray) -> np.ndarray:
    plus = np.zeros(len(gradient_true), bool)
    plus[1:20] = True
    plus[39:58] = True
    pair_plus = np.flatnonzero(plus)
    pair_minus = partner[pair_plus]
    true_pair = cosine_rows(gradient_true[pair_plus], gradient_true[pair_minus])
    return true_pair < 0.0


def h_pair_mask(centered_h: np.ndarray, partner: np.ndarray) -> np.ndarray:
    plus = np.zeros(len(centered_h), bool)
    plus[1:20] = True
    plus[39:58] = True
    pair_plus = np.flatnonzero(plus)
    pair_minus = partner[pair_plus]
    h_pair = cosine_rows(centered_h[pair_plus], centered_h[pair_minus])
    return h_pair < 0.0


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.cv_summary.read_text())
    targeted = load_npz(args.targeted_labels)
    manifest = json.loads(args.manifest.read_text())
    if sha256_file(args.targeted_labels) != summary["sources"]["targeted_labels_sha256"]:
        raise AssertionError("targeted label hash mismatch vs CV summary")
    if sha256_file(args.manifest) != summary["sources"]["manifest_sha256"]:
        raise AssertionError("manifest hash mismatch vs CV summary")
    states, row_state = build_state_table(manifest, targeted)
    stratum_by_row = {
        row: states[row_state[row]]["context_strata"][
            states[row_state[row]]["context_rows"].index(row)
        ]
        for row in row_state
    }

    initial_path = Path(summary["sources"]["initial_actor"])
    initial = torch.load(initial_path, map_location="cpu")
    alpha_payload = torch.load(initial["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial["base_move_threshold"]), args.batch_size, device,
    )
    inputs = list(actor_inputs(tensors, alpha_center, device))
    base_labels = load_npz(Path(summary["sources"]["labels"]))
    action_by_context = np.zeros((len(data.episodes), 8, 2), np.float32)
    assigned = np.zeros(len(data.episodes), bool)
    for position, context in enumerate(base_labels["context_index"].astype(np.int64)):
        action_by_context[context] = base_labels["actor_center"][position]
        assigned[context] = True
    if not np.all(assigned):
        raise AssertionError("absolute action map is incomplete")
    inputs[3] = torch.from_numpy(action_by_context).to(device)
    inputs = tuple(inputs)

    location_count = targeted["outer_centers"].shape[1]
    partner = location_partner(location_count)
    records_out = []
    per_seed_pool: dict[str, list[dict]] = {}
    checkpoint_hash_checks = {}
    for record in summary["records"]:
        checkpoint = Path(record["checkpoint"])
        checkpoint_hash_checks[
            f"fold{record['fold']}_seed{record['seed']}"
        ] = sha256_file(checkpoint) == record["checkpoint_sha256"]
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
        heldout_states = set(record["heldout_states"])
        rows = np.sort([
            row for row in row_state if row_state[row] in heldout_states
        ])
        if len(rows) != record["heldout_context_count"]:
            raise AssertionError("heldout context reconstruction mismatch")
        context_per_location = np.repeat(
            targeted["context_index"][rows].astype(np.int64), location_count
        )
        absolute_per_location = targeted["outer_centers"][rows].reshape(-1, 8, 2)
        gradient_pred, hessian = predict_locations(
            model, inputs, context_per_location, absolute_per_location,
            args.batch_size, device,
        )
        gradient_pred = gradient_pred.reshape(len(rows), location_count, 16)
        hessian = hessian.reshape(len(rows), location_count, 16, 16)
        gradient_true = targeted["local_gradient"][rows]
        local_actions = targeted["outer_actions"][rows].reshape(len(rows), location_count, 16)
        absolute_cosine, g0_cosine, g0_ratio = [], [], []
        centered_cosine, centered_ratio = [], []
        cf_h, cf_g0, cf_full, cf_full_ratio = [], [], [], []
        true_reversal, h_reversal = [], []
        for index in range(len(rows)):
            block = response_block(
                gradient_true[index], gradient_pred[index],
                hessian[index, 0], local_actions[index], partner,
            )
            absolute_cosine.append(block["absolute_cosine"])
            g0_cosine.append(block["g0_center_cosine"])
            g0_ratio.append(block["g0_norm_ratio"])
            centered_cosine.append(block["centered_cosine"])
            centered_ratio.append(block["centered_norm_ratio"])
            cf_h.append(block["cf_true_g0_plus_pred_h_cosine"])
            cf_g0.append(block["cf_pred_g0_plus_true_delta_cosine"])
            cf_full.append(block["cf_pred_g0_plus_pred_h_cosine"])
            cf_full_ratio.append(block["cf_full_norm_ratio"])
            true_reversal.append(block["true_pair_reversal"])
            h_reversal.append(block["h_pair_reversal_pred"])
        absolute_cosine = np.concatenate(absolute_cosine)
        centered_cosine = np.concatenate(centered_cosine)
        centered_ratio = np.concatenate(centered_ratio)
        cf_h = np.concatenate(cf_h)
        cf_g0 = np.concatenate(cf_g0)
        cf_full = np.concatenate(cf_full)
        cf_full_ratio = np.concatenate(cf_full_ratio)
        g0_cosine = np.asarray(g0_cosine)
        g0_ratio = np.asarray(g0_ratio)
        true_reversal = np.concatenate(true_reversal)
        h_reversal = np.concatenate(h_reversal)
        strata = np.asarray([stratum_by_row[int(row)] for row in rows])
        raw = {
            "strata_rows": strata,
            "absolute_cosine": absolute_cosine,
            "g0_cosine": g0_cosine,
            "g0_ratio": g0_ratio,
            "centered_cosine": centered_cosine,
            "centered_ratio": centered_ratio,
            "cf_true_g0_plus_pred_h": cf_h,
            "cf_pred_g0_plus_true_delta": cf_g0,
            "cf_pred_g0_plus_pred_h": cf_full,
            "cf_full_ratio": cf_full_ratio,
            "true_reversal": true_reversal,
            "h_reversal": h_reversal,
        }
        by_stratum = {}
        for stratum in np.unique(strata):
            absolute_mask = np.repeat(strata == stratum, location_count)
            outer_mask = np.repeat(strata == stratum, location_count - 1)
            by_stratum[stratum] = {
                "absolute_cosine": summarize(absolute_cosine[absolute_mask]),
                "centered_cosine": summarize(centered_cosine[outer_mask]),
                "cf_true_g0_plus_pred_h": summarize(cf_h[outer_mask]),
                "cf_pred_g0_plus_true_delta": summarize(cf_g0[outer_mask]),
                "cf_pred_g0_plus_pred_h": summarize(cf_full[outer_mask]),
            }
        entry = {
            "fold": int(record["fold"]),
            "seed": int(record["seed"]),
            "context_rows": rows.tolist(),
            "absolute_cosine": summarize(absolute_cosine),
            "g0_center": {
                **summarize(g0_cosine),
                "norm_ratio_median": float(np.median(g0_ratio)),
            },
            "centered_h_response": {
                **summarize(centered_cosine),
                "norm_ratio_median": float(np.median(centered_ratio)),
            },
            "counterfactuals": {
                "true_g0_plus_pred_h": summarize(cf_h),
                "pred_g0_plus_true_delta": summarize(cf_g0),
                "pred_g0_plus_pred_h": summarize(cf_full),
                "full_model_norm_ratio_median": float(np.median(cf_full_ratio)),
            },
            "h_only_reversal_recall": (
                float(np.mean(h_reversal[true_reversal]))
                if np.any(true_reversal) else None
            ),
            "h_only_any_flip_fraction": float(np.mean(h_reversal)),
            "by_stratum": by_stratum,
        }
        records_out.append(entry)
        per_seed_pool.setdefault(str(record["seed"]), []).append((entry, raw))
        del model
    if not all(checkpoint_hash_checks.values()):
        raise AssertionError("checkpoint hash mismatch")

    def pooled_block(raw_list: list[dict]) -> dict:
        strata_rows = np.concatenate([item["strata_rows"] for item in raw_list])
        strata_locations = np.repeat(strata_rows, location_count)
        strata_outer = np.repeat(strata_rows, location_count - 1)
        block = {
            "run_count": len(raw_list),
            "context_count": int(len(strata_rows)),
            "absolute_cosine": summarize(np.concatenate([
                item["absolute_cosine"] for item in raw_list
            ])),
            "g0_center": {
                **summarize(np.concatenate([
                    item["g0_cosine"] for item in raw_list
                ])),
                "norm_ratio_median": float(np.median(np.concatenate([
                    item["g0_ratio"] for item in raw_list
                ]))),
            },
            "centered_h_response": {
                **summarize(np.concatenate([
                    item["centered_cosine"] for item in raw_list
                ])),
                "norm_ratio_median": float(np.median(np.concatenate([
                    item["centered_ratio"] for item in raw_list
                ]))),
            },
            "counterfactuals": {
                "true_g0_plus_pred_h": summarize(np.concatenate([
                    item["cf_true_g0_plus_pred_h"] for item in raw_list
                ])),
                "pred_g0_plus_true_delta": summarize(np.concatenate([
                    item["cf_pred_g0_plus_true_delta"] for item in raw_list
                ])),
                "pred_g0_plus_pred_h": summarize(np.concatenate([
                    item["cf_pred_g0_plus_pred_h"] for item in raw_list
                ])),
                "full_model_norm_ratio_median": float(np.median(
                    np.concatenate([item["cf_full_ratio"] for item in raw_list])
                )),
            },
        }
        true_reversal = np.concatenate([item["true_reversal"] for item in raw_list])
        h_reversal = np.concatenate([item["h_reversal"] for item in raw_list])
        block["h_only_reversal_recall"] = (
            float(np.mean(h_reversal[true_reversal]))
            if np.any(true_reversal) else None
        )
        # pair_plus enumerates each antithetic pair once; no halving.
        block["true_reversal_pair_count"] = int(np.sum(true_reversal))
        block["by_stratum"] = {}
        for stratum in np.unique(strata_rows):
            location_mask = strata_locations == stratum
            outer_mask = strata_outer == stratum
            g0_mask = strata_rows == stratum
            block["by_stratum"][stratum] = {
                "absolute_cosine": summarize(np.concatenate([
                    item["absolute_cosine"] for item in raw_list
                ])[location_mask]),
                "g0_center": summarize(np.concatenate([
                    item["g0_cosine"] for item in raw_list
                ])[g0_mask]),
                "centered_cosine": summarize(np.concatenate([
                    item["centered_cosine"] for item in raw_list
                ])[outer_mask]),
                "cf_true_g0_plus_pred_h": summarize(np.concatenate([
                    item["cf_true_g0_plus_pred_h"] for item in raw_list
                ])[outer_mask]),
                "cf_pred_g0_plus_true_delta": summarize(np.concatenate([
                    item["cf_pred_g0_plus_true_delta"] for item in raw_list
                ])[outer_mask]),
                "cf_pred_g0_plus_pred_h": summarize(np.concatenate([
                    item["cf_pred_g0_plus_pred_h"] for item in raw_list
                ])[outer_mask]),
            }
        return block

    pooled_seeds = {}
    for seed, items in sorted(per_seed_pool.items()):
        rows = np.concatenate([entry["context_rows"] for entry, _ in items])
        if len(rows) != 79 or len(set(rows.tolist())) != 79:
            raise AssertionError(f"seed {seed} OOF contexts do not partition 79")
        pooled_seeds[seed] = pooled_block([raw for _, raw in items])
    # Pool across all 15 runs: each context then counted 3x (once per seed).
    pooled_all = pooled_block([
        raw for items in per_seed_pool.values() for _, raw in items
    ])
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GROUPED_CV_OOF_DECOMPOSITION_VALIDATED",
        "sources": {
            "cv_summary": str(args.cv_summary.resolve()),
            "cv_summary_sha256": sha256_file(args.cv_summary),
            "targeted_labels_sha256": sha256_file(args.targeted_labels),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "checkpoint_hash_checks": checkpoint_hash_checks,
        "records": records_out,
        "pooled_per_seed": pooled_seeds,
        "pooled_all_15_runs": pooled_all,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "zero_training_recomputation": True,
        },
    }
    output = args.output or args.cv_summary.with_name("oof_decomposition_validation.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "hash_checks_all_pass": all(checkpoint_hash_checks.values()),
        "per_run_g0_and_centered_medians": [
            {
                "fold": entry["fold"], "seed": entry["seed"],
                "g0_cosine_median": entry["g0_center"]["median"],
                "centered_cosine_median": entry["centered_h_response"]["median"],
                "h_only_reversal_recall": entry["h_only_reversal_recall"],
            }
            for entry in records_out
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
