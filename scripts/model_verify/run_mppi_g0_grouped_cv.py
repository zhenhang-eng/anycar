#!/usr/bin/env python3
"""g0-focused grouped cross-validation over all hard strata.

Protocol (review doc §11.27.1 next step):
- Consumes the 600 internal-selection fresh-FD contexts for g0 training with a
  grouped split: 5 folds over the physical states of the 204 hard contexts
  (JOINT_G0_H 34 + H_ONLY 45 + G0_ONLY_CRITIC_HARD 125), grouped by
  (episode, physical_snapshot_ordinal). All frames of a heldout state are
  held out; control-state frames are never trained.
- H is set aside: H heads (diagonal/low-rank) are excluded from the optimizer
  and stay zero-initialized; all H-dependent losses (chord, probe value) are
  dropped. Standing H metrics are not applicable this round by design.
- Arms: PA_G0 (existing semantic encoder) vs PA_G0_X (explicit state-by-action
  bilinear interaction encoder), 3 seeds per fold.
- Gates: out-of-fold g0 cosine median >= 0.70, P10 >= 0, norm ratio median in
  [0.5, 2.0]; >= 4/5 folds (>= 2/3 seeds per fold), >= 2/3 seeds pooled,
  controls not degraded vs the old PA (median paired delta >= -0.02).
- Checkpoints are selected only by the internal-validation split.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPIDeterministicCenterActor,
    TorchMPPISemanticInteractionStructuredLocalQCritic,
    TorchMPPISemanticStructuredLocalQCritic,
)
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_local_gradient_critic import cosine_rows
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_structured_local_q_critic import (
    DEFAULT_FRESH,
    DEFAULT_INITIAL,
    DEFAULT_LABELS,
    batch_parameters,
    fresh_metrics,
    load_npz,
    set_seed,
    subset_first_axis,
)

DEFAULT_TARGETED_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_BASELINE_SUMMARY = Path(
    "outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/g0_grouped_cv_20260814_v1")
HARD_STRATA = ("JOINT_G0_H", "H_ONLY", "G0_ONLY_CRITIC_HARD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--labels-npz", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--targeted-labels-npz", type=Path, default=DEFAULT_TARGETED_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", default="PA_G0,PA_G0_X")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--selection-min-epoch", type=int, default=80)
    parser.add_argument("--scheduler-min-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hessian-scale", type=float, default=256.0)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.50)
    parser.add_argument("--norm-weight", type=float, default=0.20)
    parser.add_argument("--small-chord-sigma", type=float, default=0.15)
    parser.add_argument("--medium-chord-sigma", type=float, default=0.30)
    parser.add_argument("--fold-seed", type=int, default=260814)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


HESSIAN_HEAD_PREFIXES = (
    "diagonal_head", "low_rank_vector_head", "low_rank_value_head",
)


def build_model(arm: str, args: argparse.Namespace, device: torch.device):
    if arm == "PA_G0":
        model = TorchMPPISemanticStructuredLocalQCritic(
            include_feedback=False, low_rank=2, dropout=args.dropout,
            hessian_scale=args.hessian_scale, hessian_enabled=True,
        )
    elif arm == "PA_G0_X":
        model = TorchMPPISemanticInteractionStructuredLocalQCritic(
            include_feedback=False, low_rank=2, dropout=args.dropout,
            hessian_scale=args.hessian_scale, hessian_enabled=True,
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    return model.to(device)


def initialize_from_actor(model, actor) -> None:
    model.encoder.history_encoder.load_state_dict(
        actor.encoder.history_encoder.state_dict(), strict=True
    )
    model.encoder.reference_encoder.load_state_dict(
        actor.encoder.reference_encoder.state_dict(), strict=True
    )
    model.encoder.current_encoder.load_state_dict(
        actor.encoder.current_encoder.state_dict(), strict=True
    )
    model.encoder.absolute_action_encoder.load_state_dict(
        actor.encoder.anchor_encoder.state_dict(), strict=True
    )


def g0_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    g0_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(HESSIAN_HEAD_PREFIXES)
    ]
    frozen = [
        name for name, _ in model.named_parameters()
        if name.startswith(HESSIAN_HEAD_PREFIXES)
    ]
    assert frozen, "H heads must exist and stay frozen"
    return torch.optim.AdamW(
        g0_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )


def center_metrics(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: np.ndarray,
    target_gradient: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predicted = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            local = torch.from_numpy(
                np.asarray(context[start : start + batch_size])
            ).to(device)
            _, gradient, _ = batch_parameters(model, inputs, local)
            predicted.append(gradient.cpu().numpy())
    predicted = np.concatenate(predicted).astype(np.float32)
    cosine = cosine_rows(predicted, target_gradient)
    ratio = np.linalg.norm(predicted, axis=1) / (
        np.linalg.norm(target_gradient, axis=1) + 1e-12
    )
    return {
        "count": int(len(context)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "norm_ratio_median": float(np.median(ratio)),
    }


def train_g0_fold(
    arm: str,
    fold: int,
    seed: int,
    actor,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    fresh_train: dict[str, np.ndarray],
    targeted_center: dict[str, np.ndarray],
    scales: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
):
    set_seed(seed * 100 + fold)
    model = build_model(arm, args, device)
    initialize_from_actor(model, actor)
    optimizer = g0_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.35, patience=12, min_lr=3e-7
    )
    rng = np.random.default_rng(260814 + seed * 100 + fold)
    target_value = torch.from_numpy(labels["value"]).to(device)
    target_gradient = torch.from_numpy(labels["gradient"]).to(device)
    value_scale = float(scales["value"])
    gradient_scale = torch.from_numpy(
        np.asarray(scales["gradient"], np.float32)
    ).to(device)
    fresh_context = torch.from_numpy(
        fresh_train["context_index"]
    ).to(device)
    fresh_gradient = torch.from_numpy(fresh_train["gradient"]).to(device)
    center_context = torch.from_numpy(
        targeted_center["context_index"]
    ).to(device)
    center_value = torch.from_numpy(targeted_center["value"]).to(device)
    center_gradient = torch.from_numpy(targeted_center["gradient"]).to(device)
    validation_context = labels["context_index"][validation_positions]
    validation_gradient = labels["gradient"][validation_positions]
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        base_losses = []
        order = rng.permutation(train_positions)
        for start in range(0, len(order), args.batch_size):
            pos_np = order[start : start + args.batch_size]
            pos = torch.from_numpy(pos_np).to(device)
            context = torch.from_numpy(
                labels["context_index"][pos_np]
            ).to(device)
            q0, g0, _ = batch_parameters(model, inputs, context)
            tv, tg = target_value[pos], target_gradient[pos]
            value_loss = F.smooth_l1_loss(
                (q0 - tv) / value_scale, torch.zeros_like(q0), beta=0.5
            )
            gradient_loss = F.smooth_l1_loss(
                (g0 - tg) / gradient_scale, torch.zeros_like(g0), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(g0, tg, dim=1)).mean()
            norm_loss = F.smooth_l1_loss(
                torch.log(torch.linalg.vector_norm(g0, dim=1) + 1e-4),
                torch.log(torch.linalg.vector_norm(tg, dim=1) + 1e-4),
                beta=0.5,
            )
            loss = (
                args.value_weight * value_loss
                + args.gradient_weight * gradient_loss
                + args.cosine_weight * cosine_loss
                + args.norm_weight * norm_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            base_losses.append(float(loss.detach()))
        fresh_losses = []
        fresh_order = rng.permutation(len(fresh_context))
        for start in range(0, len(fresh_order), args.batch_size):
            local = fresh_order[start : start + args.batch_size]
            index = torch.from_numpy(local).to(device)
            context = fresh_context[index]
            _, g0, _ = batch_parameters(model, inputs, context)
            tg = fresh_gradient[index]
            gradient_loss = F.smooth_l1_loss(
                (g0 - tg) / gradient_scale, torch.zeros_like(g0), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(g0, tg, dim=1)).mean()
            norm_loss = F.smooth_l1_loss(
                torch.log(torch.linalg.vector_norm(g0, dim=1) + 1e-4),
                torch.log(torch.linalg.vector_norm(tg, dim=1) + 1e-4),
                beta=0.5,
            )
            loss = (
                args.gradient_weight * gradient_loss
                + args.cosine_weight * cosine_loss
                + args.norm_weight * norm_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            fresh_losses.append(float(loss.detach()))
        center_losses = []
        center_order = rng.permutation(len(center_context))
        for start in range(0, len(center_order), args.batch_size):
            local = center_order[start : start + args.batch_size]
            index = torch.from_numpy(local).to(device)
            context = center_context[index]
            q0, g0, _ = batch_parameters(model, inputs, context)
            tv, tg = center_value[index], center_gradient[index]
            value_loss = F.smooth_l1_loss(
                (q0 - tv) / value_scale, torch.zeros_like(q0), beta=0.5
            )
            gradient_loss = F.smooth_l1_loss(
                (g0 - tg) / gradient_scale, torch.zeros_like(g0), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(g0, tg, dim=1)).mean()
            norm_loss = F.smooth_l1_loss(
                torch.log(torch.linalg.vector_norm(g0, dim=1) + 1e-4),
                torch.log(torch.linalg.vector_norm(tg, dim=1) + 1e-4),
                beta=0.5,
            )
            loss = (
                args.value_weight * value_loss
                + args.gradient_weight * gradient_loss
                + args.cosine_weight * cosine_loss
                + args.norm_weight * norm_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            center_losses.append(float(loss.detach()))
        validation = center_metrics(
            model, inputs, validation_context, validation_gradient,
            args.evaluation_batch_size, device,
        )
        score = (
            1.0 - validation["cosine_median"]
            + 0.5 * (1.0 - validation["cosine_p10"])
            + 0.2 * abs(np.log(validation["norm_ratio_median"] + 1e-12))
        )
        if epoch >= args.scheduler_min_epoch:
            scheduler.step(score)
        eligible = (
            epoch >= args.selection_min_epoch
            and 0.5 <= validation["norm_ratio_median"] <= 2.0
        )
        if eligible and score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif eligible:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            history.append({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "checkpoint_eligible": eligible,
                "score": float(score),
                "base_loss": float(np.mean(base_losses)),
                "fresh_loss": float(np.mean(fresh_losses)) if fresh_losses else None,
                "center_loss": float(np.mean(center_losses)) if center_losses else None,
                "validation": validation,
            })
            print(json.dumps({
                "arm": arm, "fold": fold, "seed": seed, **history[-1],
            }), flush=True)
        if eligible and stale >= args.patience:
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_score = score
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "arm": arm, "fold": int(fold), "seed": int(seed),
        "best_epoch": int(best_epoch), "epochs_run": int(epoch),
        "best_validation_score": float(best_score), "history": history,
    }


def gate_block(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "cosine_median_ge_0_70": metrics["cosine_median"] >= 0.70,
        "cosine_p10_ge_0": metrics["cosine_p10"] >= 0.0,
        "norm_ratio_in_0_5_2_0": (
            0.5 <= metrics["norm_ratio_median"] <= 2.0
        ),
    }


def main() -> None:
    args = parse_args()
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = list(actor_inputs(tensors, alpha_center, device))
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()

    labels = load_npz(args.labels_npz)
    labels["gradient"] = labels["gradient_by_radius"][:, 0].astype(np.float32)
    labels["value"] = labels["transformed_reward"][:, 0, 0].astype(np.float32)
    context = labels["context_index"].astype(np.int64)
    actor_center_by_context = np.zeros((len(data.episodes), 8, 2), np.float32)
    assigned = np.zeros(len(data.episodes), bool)
    for position, value in enumerate(context):
        actor_center_by_context[value] = labels["actor_center"][position]
        assigned[value] = True
    if not np.all(assigned):
        raise AssertionError("absolute action map is incomplete")
    inputs[3] = torch.from_numpy(actor_center_by_context).to(device)
    inputs = tuple(inputs)
    fit_positions = np.flatnonzero(np.isin(
        data.episodes[context], splits["internal_fit"]
    ))
    fit_episodes = np.unique(data.episodes[context[fit_positions]])
    split_rng = np.random.default_rng(260812)
    shuffled = fit_episodes.copy()
    split_rng.shuffle(shuffled)
    validation_episodes = set(shuffled[: max(1, len(shuffled) // 5)])
    validation_positions = fit_positions[np.asarray([
        episode in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    train_positions = fit_positions[np.asarray([
        episode not in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    scales = {
        "value": max(float(np.std(labels["value"][train_positions])), 0.05),
        "gradient": np.maximum(
            np.std(labels["gradient"][train_positions], axis=0), 0.05
        ).astype(np.float32),
    }

    fresh = load_npz(args.fresh_npz)
    manifest = json.loads(args.manifest.read_text())
    fresh_context = fresh["context_index"].astype(np.int64)
    if set(fresh_context.tolist()) != {
        int(row["context_index"]) for row in manifest["rows"]
    }:
        raise AssertionError("fresh frames and manifest contexts mismatch")
    row_by_context = {
        int(row["context_index"]): row for row in manifest["rows"]
    }
    fresh_state = np.asarray([
        f"{row_by_context[int(value)]['episode']}#"
        f"{int(row_by_context[int(value)]['physical_snapshot_ordinal'])}"
        for value in fresh_context
    ])
    fresh_stratum = np.asarray([
        row_by_context[int(value)]["stratum_id"] for value in fresh_context
    ])
    fresh_scenario = np.asarray([
        row_by_context[int(value)]["scenario"] for value in fresh_context
    ])

    targeted = load_npz(args.targeted_labels_npz)
    targeted_validation = json.loads(
        (args.targeted_labels_npz.parent / "validation_summary.json").read_text()
    )
    if targeted_validation["qualification"] != "TARGETED_LOCAL_RESPONSE_LABELS_VALIDATED":
        raise AssertionError("targeted response labels are not validated")
    control_rows = np.flatnonzero(targeted["pilot_role"] == "matched_easy_control")
    target_rows = np.flatnonzero(targeted["pilot_role"] == "target")
    if len(control_rows) != 21 or len(target_rows) != 79:
        raise AssertionError("target/control count mismatch")
    control_states = {
        f"{row_by_context[int(targeted['context_index'][row])]['episode']}#"
        f"{int(row_by_context[int(targeted['context_index'][row])]['physical_snapshot_ordinal'])}"
        for row in control_rows
    }
    # Hard states: every physical state owning a hard-stratum fresh frame.
    hard_mask = np.isin(fresh_stratum, HARD_STRATA)
    hard_states = {}
    for position in np.flatnonzero(hard_mask):
        key = fresh_state[position]
        entry = hard_states.setdefault(key, {
            "state_key": key,
            "strata": set(), "scenario": fresh_scenario[position],
            "frames": [],
        })
        entry["strata"].add(fresh_stratum[position])
        entry["frames"].append(position)
    if len(hard_states) < 50:
        raise AssertionError(f"unexpected hard state count {len(hard_states)}")

    def state_stratum(entry: dict) -> str:
        # Worst-case lift for fold balancing; per-frame strata stay the
        # reporting unit.
        for stratum in HARD_STRATA:
            if stratum in entry["strata"]:
                return stratum
        raise AssertionError("hard state without hard stratum")

    for entry in hard_states.values():
        entry["stratum_id"] = state_stratum(entry)
        entry["repeat_count"] = len(entry["frames"])

    rng = np.random.default_rng(args.fold_seed)
    keys = sorted(hard_states)
    keys = [keys[index] for index in rng.permutation(len(keys))]
    ordered = sorted(
        keys, key=lambda key: HARD_STRATA.index(hard_states[key]["stratum_id"])
    )
    assignment: dict[str, int] = {}
    for key in ordered:
        meta = hard_states[key]
        scores = []
        for fold in range(args.folds):
            members = [
                name for name, owner in assignment.items() if owner == fold
            ]
            scores.append((
                sum(
                    hard_states[name]["stratum_id"] == meta["stratum_id"]
                    for name in members
                ),
                len(members),
                sum(
                    hard_states[name]["scenario"] == meta["scenario"]
                    for name in members
                ),
                sum(
                    hard_states[name]["repeat_count"] == meta["repeat_count"]
                    for name in members
                ),
            ))
        assignment[key] = min(range(args.folds), key=lambda fold: scores[fold])
    fold_states = {
        fold: sorted(
            key for key, owner in assignment.items() if owner == fold
        )
        for fold in range(args.folds)
    }

    # Training exclusion is state-level: no frame of a control state is ever
    # trained. Evaluation uses only the 21 control-role frames; the strict
    # subset drops controls whose state also owns a target context (6018/6019).
    control_state_mask = np.isin(fresh_state, sorted(control_states))
    fresh_role = np.asarray([
        row_by_context[int(value)]["pilot_role"] for value in fresh_context
    ])
    control_frames = np.flatnonzero(fresh_role == "matched_easy_control")
    if len(control_frames) != 21:
        raise AssertionError("control fresh frames missing")
    target_state_keys = {
        f"{row_by_context[int(targeted['context_index'][row])]['episode']}#"
        f"{int(row_by_context[int(targeted['context_index'][row])]['physical_snapshot_ordinal'])}"
        for row in target_rows
    }
    strict_control_frames = np.asarray([
        frame for frame in control_frames
        if fresh_state[frame] not in target_state_keys
    ], np.int64)
    if len(strict_control_frames) != 20:
        raise AssertionError("strict control count mismatch")

    consumed_episode = set(
        fresh["episode"][np.isin(
            fresh_context, targeted["context_index"][target_rows].astype(np.int64)
        )].tolist()
    )
    unseen_positions = np.asarray([
        position for position, value in enumerate(fresh["episode"])
        if value not in consumed_episode
    ], np.int64)
    fresh_unseen = subset_first_axis(fresh, unseen_positions)

    baseline_summary = json.loads(args.baseline_summary.read_text())
    if baseline_summary["sources"]["initial_actor_sha256"] != sha256_file(args.initial_actor):
        raise AssertionError("baseline PA used a different frozen Actor")
    baseline_records = {
        int(record["seed"]): record
        for record in baseline_summary["arms"]["PA"]["records"]
    }
    baseline_control: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        record = baseline_records[seed]
        checkpoint = Path(record["checkpoint"])
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise AssertionError(f"baseline seed{seed} checkpoint hash mismatch")
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
        baseline_control[str(seed)] = {
            "control_21": center_metrics(
                model, inputs, fresh_context[control_frames],
                fresh["gradient"][control_frames].astype(np.float32),
                args.evaluation_batch_size, device,
            ),
            "control_20_state_strict": center_metrics(
                model, inputs, fresh_context[strict_control_frames],
                fresh["gradient"][strict_control_frames].astype(np.float32),
                args.evaluation_batch_size, device,
            ),
        }
        del model

    records: list[dict[str, Any]] = []
    for arm in arms:
        for fold in range(args.folds):
            heldout = set(fold_states[fold])
            heldout_frames = np.sort(np.concatenate([
                hard_states[key]["frames"] for key in fold_states[fold]
            ]))
            fresh_train_mask = ~np.isin(fresh_state, sorted(heldout)) & ~control_state_mask
            fresh_train = {
                "context_index": fresh_context[fresh_train_mask],
                "gradient": fresh["gradient"][fresh_train_mask].astype(np.float32),
            }
            targeted_center = {
                "context_index": np.asarray([], np.int64),
                "value": np.asarray([], np.float32),
                "gradient": np.asarray([], np.float32),
            }
            center_rows = [
                row for row in target_rows
                if f"{row_by_context[int(targeted['context_index'][row])]['episode']}#"
                f"{int(row_by_context[int(targeted['context_index'][row])]['physical_snapshot_ordinal'])}"
                not in heldout
            ]
            targeted_center = {
                "context_index": targeted["context_index"][
                    center_rows
                ].astype(np.int64),
                "value": targeted["transformed_reward"][
                    center_rows, 0, 0
                ].astype(np.float32),
                "gradient": targeted["local_gradient"][
                    center_rows, 0
                ].astype(np.float32),
            }
            for seed in seeds:
                model, training = train_g0_fold(
                    arm, fold, seed, actor, inputs, labels, train_positions,
                    validation_positions, fresh_train, targeted_center,
                    scales, args, device,
                )
                checkpoint_path = (
                    args.output_dir / f"g0_{arm}_fold{fold}_seed{seed}.pt"
                )
                torch.save({
                    "format_version": 1,
                    "model_class": (
                        "TorchMPPISemanticInteractionStructuredLocalQCritic"
                        if arm == "PA_G0_X" else
                        "TorchMPPISemanticStructuredLocalQCritic"
                    ),
                    "model_state_dict": model.state_dict(),
                    "arm": arm, "fold": int(fold), "seed": int(seed),
                    "heldout_states": sorted(heldout),
                    "hessian_frozen_zero": True,
                    "initial_actor": str(args.initial_actor.resolve()),
                    "initial_actor_sha256": sha256_file(args.initial_actor),
                    "training": training,
                    "contract": {
                        "actor_frozen": True,
                        "formal_validation_loaded": False,
                        "test_loaded": False,
                        "hessian_heads_excluded_from_optimizer": True,
                        "h_losses_dropped": True,
                        "controls_never_trained": True,
                        "checkpoint_selected_by": "internal_validation_only",
                    },
                }, checkpoint_path)
                out_of_fold = center_metrics(
                    model, inputs, fresh_context[heldout_frames],
                    fresh["gradient"][heldout_frames].astype(np.float32),
                    args.evaluation_batch_size, device,
                )
                by_stratum = {}
                for stratum in HARD_STRATA:
                    stratum_frames = heldout_frames[
                        fresh_stratum[heldout_frames] == stratum
                    ]
                    if len(stratum_frames) == 0:
                        continue
                    by_stratum[stratum] = center_metrics(
                        model, inputs, fresh_context[stratum_frames],
                        fresh["gradient"][stratum_frames].astype(np.float32),
                        args.evaluation_batch_size, device,
                    )
                control = center_metrics(
                    model, inputs, fresh_context[control_frames],
                    fresh["gradient"][control_frames].astype(np.float32),
                    args.evaluation_batch_size, device,
                )
                control_strict = center_metrics(
                    model, inputs, fresh_context[strict_control_frames],
                    fresh["gradient"][strict_control_frames].astype(np.float32),
                    args.evaluation_batch_size, device,
                )
                fresh230 = fresh_metrics(
                    model, inputs, fresh_unseen,
                    float(initial_payload["maximum_residual_sigma"]), args,
                    device,
                )
                records.append({
                    "arm": arm, "fold": int(fold), "seed": int(seed),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "heldout_states": sorted(heldout),
                    "heldout_frame_count": int(len(heldout_frames)),
                    "fresh_train_frame_count": int(fresh_train_mask.sum()),
                    "targeted_center_count": int(len(center_rows)),
                    "training": {
                        key: training[key] for key in (
                            "arm", "fold", "seed", "best_epoch", "epochs_run",
                            "best_validation_score",
                        )
                    },
                    "out_of_fold": out_of_fold,
                    "out_of_fold_by_stratum": by_stratum,
                    "control_fresh_21": control,
                    "control_fresh_20_state_strict": control_strict,
                    "control_paired_delta_vs_baseline": {
                        metric: control_strict[metric]
                        - baseline_control[str(seed)][
                            "control_20_state_strict"
                        ][metric]
                        for metric in (
                            "cosine_median", "cosine_p10", "norm_ratio_median",
                        )
                    },
                    "fresh_fd_unseen_230": fresh230,
                    "out_of_fold_gates": gate_block(out_of_fold),
                })
                del model

    decision: dict[str, Any] = {"arms": {}}
    for arm in arms:
        arm_records = [r for r in records if r["arm"] == arm]
        fold_pass = {}
        for fold in range(args.folds):
            runs = [r for r in arm_records if r["fold"] == fold]
            passing = sum(
                r["out_of_fold_gates"]["cosine_median_ge_0_70"]
                and r["out_of_fold_gates"]["norm_ratio_in_0_5_2_0"]
                for r in runs
            )
            fold_pass[str(fold)] = {
                "seed_pass_count": int(passing),
                "fold_passes": passing >= 2,
            }
        folds_passing = sum(
            entry["fold_passes"] for entry in fold_pass.values()
        )
        seed_full = {}
        for seed in seeds:
            runs = [r for r in arm_records if r["seed"] == seed]
            full = all(all(r["out_of_fold_gates"].values()) for r in runs)
            seed_full[str(seed)] = {
                "folds_with_all_gates": int(sum(
                    all(r["out_of_fold_gates"].values()) for r in runs
                )),
                "full_pass": bool(full),
            }
        seeds_passing = sum(entry["full_pass"] for entry in seed_full.values())
        paired_cosine = [
            r["control_paired_delta_vs_baseline"]["cosine_median"]
            for r in arm_records
        ]
        paired_p10 = [
            r["control_paired_delta_vs_baseline"]["cosine_p10"]
            for r in arm_records
        ]
        control_ok = (
            float(np.median(paired_cosine)) >= -0.02
            and float(np.median(paired_p10)) >= -0.02
        )
        decision["arms"][arm] = {
            "fold_pass": fold_pass,
            "folds_passing": int(folds_passing),
            "fold_rule_passed": bool(folds_passing >= 4),
            "seed_full_pass": seed_full,
            "seeds_passing": int(seeds_passing),
            "seed_rule_passed": bool(seeds_passing >= 2),
            "control_median_delta": {
                "cosine_median": float(np.median(paired_cosine)),
                "cosine_p10": float(np.median(paired_p10)),
            },
            "control_non_degradation_passed": bool(control_ok),
            "overall_pass": bool(
                folds_passing >= 4 and seeds_passing >= 2 and control_ok
            ),
        }
    any_pass = any(
        entry["overall_pass"] for entry in decision["arms"].values()
    )
    qualification = (
        "G0_GROUPED_CV_PASS_COMBINED_RETRAIN_AUTHORIZED"
        if any_pass else "G0_GROUPED_CV_FAIL_REPRESENTATION_STILL_BLOCKING"
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "labels": str(args.labels_npz.resolve()),
            "labels_sha256": sha256_file(args.labels_npz),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "targeted_labels": str(args.targeted_labels_npz.resolve()),
            "targeted_labels_sha256": sha256_file(args.targeted_labels_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_summary": str(args.baseline_summary.resolve()),
            "baseline_summary_sha256": sha256_file(args.baseline_summary),
        },
        "protocol": {
            "arms": arms,
            "folds": args.folds,
            "seeds": seeds,
            "hard_states": len(hard_states),
            "hard_frames": int(hard_mask.sum()),
            "grouping_key": "(episode, physical_snapshot_ordinal)",
            "hessian": (
                "heads excluded from optimizer, stay zero; all H-dependent "
                "losses dropped; standing H metrics not applicable this round"
            ),
            "control_frames": int(len(control_frames)),
            "control_eval": (
                "21 control-role fresh frames; gate uses the strict 20 whose "
                "state owns no target context; training exclusion is "
                "state-level across all 42 control-state frames"
            ),
            "gates": {
                "cosine_median": 0.70, "cosine_p10": 0.0,
                "norm_ratio": [0.5, 2.0],
                "folds": ">=4/5 with >=2/3 seeds",
                "seeds": ">=2/3 all-fold full pass",
                "control_median_delta": ">=-0.02",
            },
        },
        "fold_assignment": assignment,
        "fold_composition": {
            str(fold): {
                "state_count": len(fold_states[fold]),
                "frame_count": int(sum(
                    len(hard_states[key]["frames"])
                    for key in fold_states[fold]
                )),
                "strata": {
                    stratum: sum(
                        hard_states[key]["stratum_id"] == stratum
                        for key in fold_states[fold]
                    )
                    for stratum in HARD_STRATA
                },
            }
            for fold in range(args.folds)
        },
        "baseline_control_metrics": baseline_control,
        "records": records,
        "decision": {
            **decision,
            "authorization_note": (
                "Pass authorizes a combined g0+H retrain with standing H "
                "magnitude/tail gates re-enabled; it does not unfreeze the "
                "Actor. formal validation/test remain sealed."
            ),
        },
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": qualification,
        "arms": {
            arm: {
                "folds_passing": entry["folds_passing"],
                "seeds_passing": entry["seeds_passing"],
                "control_ok": entry["control_non_degradation_passed"],
                "overall_pass": entry["overall_pass"],
            }
            for arm, entry in decision["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
