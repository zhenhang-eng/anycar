#!/usr/bin/env python3
"""Value-delta grouped CV pilot v2: true scalar Q(s,a), relative supervision.

Rerun contract (review of the invalidated first attempt):
- Q(s, a) is the scalar value head over the semantic encoder: encoder -> trunk
  -> value_head. The g0/H heads are dead weight in this pilot (never used for
  prediction or loss); gradients are derived by autograd dQ/da at a0.
- Supervision is relative value Delta Q_i = Q(s, a_i) - Q(s, a_0) from the
  33-point probe labels; fresh primary radius 0.02 sigma, 0.01/0.04 are
  consistency-only, base pool uses its smallest radius 0.05 sigma.
- Batch alignment is unit-tested before any training: batched output equals
  per-sample output, Delta Q == 0 when a_i == a_0, permutation consistency.
- Checkpoints are selected by value-delta metrics (inner grouped hard
  validation + easy internal validation); gradient metrics are derived
  evaluation only.
- Records include the strict-20 control frames, per-seed strict pooled
  statistics, and complete gates.
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
    load_npz,
    set_seed,
    subset_first_axis,
)
from run_mppi_g0_grouped_cv import HARD_STRATA

DEFAULT_TARGETED_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_GRADIENT_BASELINE = Path(
    "outputs/mppi_proposal/g0_balanced_grouped_cv_20260814_v1/summary.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/g0_value_delta_cv_20260817_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--labels-npz", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--targeted-labels-npz", type=Path, default=DEFAULT_TARGETED_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gradient-baseline", type=Path, default=DEFAULT_GRADIENT_BASELINE)
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
    parser.add_argument("--small-chord-sigma", type=float, default=0.15)
    parser.add_argument("--medium-chord-sigma", type=float, default=0.30)
    parser.add_argument("--batches-per-pool", type=int, default=8)
    parser.add_argument("--hard-validation-state-fraction", type=float, default=0.2)
    parser.add_argument("--primary-radius-index", type=int, default=1)
    parser.add_argument("--base-radius-index", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=260814)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_model(arm: str, args, device):
    from run_mppi_g0_grouped_cv import build_model as _build_model
    return _build_model(arm, args, device)


def initialize_from_actor(model, actor):
    from run_mppi_g0_grouped_cv import (
        initialize_from_actor as _initialize_from_actor,
    )
    _initialize_from_actor(model, actor)


def value_optimizer(model, args):
    # Only the scalar-value path parameters (encoder + trunk + value head).
    # Hessian heads are dead in this pilot and excluded; the g0 head is dead
    # too but shares the trunk, so it is excluded as well.
    prefixes = (
        "diagonal_head", "low_rank_vector_head", "low_rank_value_head",
        "gradient_head",
    )
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(tuple(prefixes))
    ]
    return torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )


def scalar_q(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    absolute_action: torch.Tensor,
) -> torch.Tensor:
    feature = model.trunk(model.encoder(
        inputs[0][context], inputs[1][context], inputs[2][context],
        absolute_action, inputs[4][context], inputs[5][context],
    ))
    return model.value_head(feature).squeeze(-1)


def probe_delta_value(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    reference_action: torch.Tensor,
    absolute_action: torch.Tensor,
) -> torch.Tensor:
    """Q(s, a_i) - Q(s, a_0) with strictly row-aligned stacking."""
    batch = context.shape[0]
    stacked_context = torch.cat((context, context), dim=0)
    stacked_action = torch.cat((reference_action, absolute_action), dim=0)
    value = scalar_q(model, inputs, stacked_context, stacked_action)
    return value[batch:] - value[:batch]


def q_gradient(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    reference_action: torch.Tensor,
) -> torch.Tensor:
    """Derived gradient dQ/da at a0 (16-D flattened)."""
    action = reference_action.detach().clone().requires_grad_(True)
    value = scalar_q(model, inputs, context, action)
    gradient = torch.autograd.grad(value.sum(), action)[0]
    return gradient.reshape(len(context), 16)


@torch.no_grad()
def run_unit_tests(model, inputs, device) -> dict[str, bool]:
    rng = np.random.default_rng(0)
    contexts = torch.from_numpy(
        rng.choice(400, 5, replace=False).astype(np.int64)
    ).to(device)
    reference = torch.randn(5, 8, 2, device=device)
    absolute = reference + 0.05 * torch.randn(5, 8, 2, device=device)
    model.eval()
    batched = probe_delta_value(model, inputs, contexts, reference, absolute)
    singles = torch.cat([
        probe_delta_value(
            model, inputs, contexts[i : i + 1],
            reference[i : i + 1], absolute[i : i + 1],
        )
        for i in range(5)
    ])
    batch_equal = bool(torch.allclose(batched, singles, atol=1e-5))
    zero = probe_delta_value(model, inputs, contexts, reference, reference)
    zero_delta = float(zero.abs().max()) == 0.0
    perm = torch.from_numpy(rng.permutation(5)).to(device)
    permuted = probe_delta_value(
        model, inputs, contexts[perm], reference[perm], absolute[perm]
    )
    permutation_ok = bool(torch.allclose(
        batched[perm], permuted, atol=1e-5
    ))
    return {
        "batch_equals_per_sample": batch_equal,
        "zero_delta_at_reference": zero_delta,
        "permutation_consistent": permutation_ok,
    }


def value_metrics(
    model,
    inputs: tuple[torch.Tensor, ...],
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    radius_index: int,
    radius_sigma: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    point_count = actions.shape[2] - 1
    predicted = np.zeros((len(contexts), point_count), np.float32)
    model.eval()
    with torch.no_grad():
        for point in range(1, actions.shape[2]):
            column = []
            for start in range(0, len(contexts), batch_size):
                stop = min(start + batch_size, len(contexts))
                local_context = torch.from_numpy(
                    np.asarray(contexts[start:stop])
                ).to(device)
                reference = torch.from_numpy(
                    np.asarray(actions[start:stop, radius_index, 0], np.float32)
                ).to(device)
                absolute = torch.from_numpy(
                    np.asarray(actions[start:stop, radius_index, point], np.float32)
                ).to(device)
                column.append(probe_delta_value(
                    model, inputs, local_context, reference, absolute,
                ).cpu().numpy())
            predicted[:, point - 1] = np.concatenate(column)
    true = rewards[:, radius_index, 1:] - rewards[:, radius_index, :1]
    error = predicted - true
    scale = float(np.std(true)) + 1e-6
    return {
        "frame_count": int(len(contexts)),
        "radius_sigma": float(radius_sigma),
        "delta_correlation": float(np.corrcoef(
            predicted.reshape(-1), true.reshape(-1)
        )[0, 1]),
        "delta_rmse_over_scale": float(
            np.sqrt(np.mean(np.square(error))) / scale
        ),
        "pair_sign_accuracy": float(np.mean(
            np.sign(predicted) == np.sign(true)
        )),
        "_predicted": predicted,
        "_true": true,
    }


def gradient_metrics(
    model,
    inputs: tuple[torch.Tensor, ...],
    contexts: np.ndarray,
    references: np.ndarray,
    target_gradient: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    predicted = []
    model.eval()
    for start in range(0, len(contexts), batch_size):
        stop = min(start + batch_size, len(contexts))
        context = torch.from_numpy(
            np.asarray(contexts[start:stop])
        ).to(device)
        reference = torch.from_numpy(
            np.asarray(references[start:stop], np.float32)
        ).to(device)
        predicted.append(q_gradient(
            model, inputs, context, reference
        ).cpu().numpy())
    predicted = np.concatenate(predicted).astype(np.float32)
    cosine = cosine_rows(predicted, target_gradient)
    ratio = np.linalg.norm(predicted, axis=1) / (
        np.linalg.norm(target_gradient, axis=1) + 1e-12
    )
    return {
        "count": int(len(contexts)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "norm_ratio_median": float(np.median(ratio)),
        "_cosine": cosine,
    }


def value_score(metrics: dict[str, Any]) -> float:
    return (
        1.0 - metrics["delta_correlation"]
        + 0.5 * metrics["delta_rmse_over_scale"]
        + 0.25 * (1.0 - metrics["pair_sign_accuracy"])
    )


def sample_value_batch(pool, batch_size, rng):
    indices = rng.integers(len(pool["context"]), size=batch_size)
    return (
        pool["context"][indices], pool["reference"][indices],
        pool["absolute"][indices], pool["true_delta"][indices],
    )


def sample_hard_value_batch(hard_sampler, batch_size, rng):
    strata = [
        stratum for stratum in hard_sampler["strata"]
        if hard_sampler["states_by_stratum"].get(stratum)
    ]
    contexts = np.empty(batch_size, np.int64)
    references = np.empty((batch_size, 8, 2), np.float32)
    absolutes = np.empty((batch_size, 8, 2), np.float32)
    trues = np.empty(batch_size, np.float32)
    for position in range(batch_size):
        stratum = strata[rng.integers(len(strata))]
        states = hard_sampler["states_by_stratum"][stratum]
        state = states[rng.integers(len(states))]
        pool = hard_sampler["pool_by_state"][state]
        index = int(rng.integers(len(pool["context"])))
        contexts[position] = pool["context"][index]
        references[position] = pool["reference"][index]
        absolutes[position] = pool["absolute"][index]
        trues[position] = pool["true_delta"][index]
    return contexts, references, absolutes, trues


def make_pool(contexts, actions, rewards, radius_index):
    point_count = actions.shape[2] - 1
    reference = actions[:, radius_index, 0].astype(np.float32)
    true_delta = (
        rewards[:, radius_index, 1:] - rewards[:, radius_index, :1]
    ).astype(np.float32)
    return {
        "context": np.repeat(contexts.astype(np.int64), point_count),
        "reference": np.repeat(
            reference[:, None], point_count, axis=1
        ).reshape(-1, 8, 2),
        "absolute": actions[:, radius_index, 1:].reshape(-1, 8, 2).astype(np.float32),
        "true_delta": true_delta.reshape(-1),
        "delta_scale": float(np.std(true_delta.reshape(-1))) + 1e-6,
    }


def train_value_fold(
    arm, fold, seed, actor, inputs,
    base_pool, easy_pool, hard_sampler,
    validation_rows, hard_val_frames, fresh, labels,
    args, device,
):
    set_seed(seed * 100 + fold)
    model = build_model(arm, args, device)
    initialize_from_actor(model, actor)
    optimizer = value_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.35, patience=12, min_lr=3e-7
    )
    rng = np.random.default_rng(260814 + seed * 100 + fold)
    validation_context = labels["context_index"][validation_rows]
    validation_actions = labels["actions"][validation_rows]
    validation_rewards = labels["transformed_reward"][validation_rows]
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = {"base": [], "easy": [], "hard": []}
        for pool_name in ("base", "easy", "hard"):
            pool = (
                base_pool if pool_name == "base"
                else easy_pool if pool_name == "easy" else None
            )
            for _ in range(args.batches_per_pool):
                if pool is not None:
                    ctx_np, ref_np, abs_np, true_np = sample_value_batch(
                        pool, args.batch_size, rng
                    )
                    pool_scale = pool["delta_scale"]
                else:
                    ctx_np, ref_np, abs_np, true_np = sample_hard_value_batch(
                        hard_sampler, args.batch_size, rng
                    )
                    pool_scale = hard_sampler["delta_scale"]
                context = torch.from_numpy(ctx_np).to(device)
                reference = torch.from_numpy(ref_np).to(device)
                absolute = torch.from_numpy(abs_np).to(device)
                true = torch.from_numpy(true_np).to(device)
                predicted = probe_delta_value(
                    model, inputs, context, reference, absolute
                )
                loss = F.smooth_l1_loss(
                    predicted / pool_scale, true / pool_scale, beta=0.5
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses[pool_name].append(float(loss.detach()))
        model.eval()
        easy_validation = value_metrics(
            model, inputs, validation_context, validation_actions,
            validation_rewards, args.base_radius_index,
            float(labels["probe_radii_sigma"][args.base_radius_index]),
            args.evaluation_batch_size, device,
        )
        hard_validation = value_metrics(
            model, inputs, fresh["context_index"][hard_val_frames],
            fresh["actions"][hard_val_frames],
            fresh["transformed_reward"][hard_val_frames],
            args.primary_radius_index, float(fresh["probe_radii_sigma"][args.primary_radius_index]),
            args.evaluation_batch_size, device,
        )
        score = value_score(easy_validation) + value_score(hard_validation)
        if epoch >= args.scheduler_min_epoch:
            scheduler.step(score)
        eligible = (
            epoch >= args.selection_min_epoch
            and easy_validation["delta_rmse_over_scale"] < 1.10
            and hard_validation["delta_rmse_over_scale"] < 1.10
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
                "loss": {k: float(np.mean(v)) for k, v in losses.items()},
                "easy_validation": {
                    k: v for k, v in easy_validation.items()
                    if not k.startswith("_")
                },
                "hard_validation": {
                    k: v for k, v in hard_validation.items()
                    if not k.startswith("_")
                },
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
    validation_rows = fit_positions[np.asarray([
        episode in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    train_rows = fit_positions[np.asarray([
        episode not in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]

    fresh = load_npz(args.fresh_npz)
    manifest = json.loads(args.manifest.read_text())
    fresh_context = fresh["context_index"].astype(np.int64)
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
    fresh_role = np.asarray([
        row_by_context[int(value)]["pilot_role"] for value in fresh_context
    ])
    targeted = load_npz(args.targeted_labels_npz)
    control_rows = np.flatnonzero(targeted["pilot_role"] == "matched_easy_control")
    target_rows = np.flatnonzero(targeted["pilot_role"] == "target")

    def state_of(context_index: int) -> str:
        row = row_by_context[int(context_index)]
        return f"{row['episode']}#{int(row['physical_snapshot_ordinal'])}"

    control_states = {
        state_of(targeted["context_index"][row]) for row in control_rows
    }
    control_state_mask = np.isin(fresh_state, sorted(control_states))
    control_frames = np.flatnonzero(fresh_role == "matched_easy_control")
    target_state_keys = {
        state_of(targeted["context_index"][row]) for row in target_rows
    }
    strict_control_frames = np.asarray([
        frame for frame in control_frames
        if fresh_state[frame] not in target_state_keys
    ], np.int64)
    if len(strict_control_frames) != 20:
        raise AssertionError("strict control count mismatch")

    hard_mask = np.isin(fresh_stratum, HARD_STRATA)
    hard_states = {}
    for position in np.flatnonzero(hard_mask):
        key = fresh_state[position]
        entry = hard_states.setdefault(key, {
            "state_key": key, "strata": set(),
            "scenario": fresh_scenario[position], "frames": [],
        })
        entry["strata"].add(fresh_stratum[position])
        entry["frames"].append(position)

    def state_stratum(entry: dict) -> str:
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
                sum(hard_states[name]["stratum_id"] == meta["stratum_id"]
                    for name in members),
                len(members),
                sum(hard_states[name]["scenario"] == meta["scenario"]
                    for name in members),
                sum(hard_states[name]["repeat_count"] == meta["repeat_count"]
                    for name in members),
            ))
        assignment[key] = min(range(args.folds), key=lambda fold: scores[fold])
    fold_states = {
        fold: sorted(key for key, owner in assignment.items() if owner == fold)
        for fold in range(args.folds)
    }
    inner_split: dict[int, tuple[list[str], list[str]]] = {}
    for fold in range(args.folds):
        heldout = set(fold_states[fold])
        trainable = [
            key for key in sorted(hard_states)
            if key not in heldout and key not in control_states
        ]
        fold_rng = np.random.default_rng(args.fold_seed + 1000 + fold)
        train_states, val_states = [], []
        for stratum in HARD_STRATA:
            pool = [
                key for key in trainable
                if hard_states[key]["stratum_id"] == stratum
            ]
            order = fold_rng.permutation(len(pool))
            val_count = int(round(
                len(pool) * args.hard_validation_state_fraction
            ))
            val_states.extend(pool[index] for index in order[:val_count])
            train_states.extend(pool[index] for index in order[val_count:])
        inner_split[fold] = (train_states, val_states)

    base_pool = make_pool(
        labels["context_index"][train_rows], labels["actions"][train_rows],
        labels["transformed_reward"][train_rows], args.base_radius_index,
    )
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

    gradient_baseline = json.loads(args.gradient_baseline.read_text())
    baseline_by_key = {
        (r["arm"], r["fold"], r["seed"]): r
        for r in gradient_baseline["records"]
    }

    # Unit tests must pass before any training.
    test_model = build_model(arms[0], args, device)
    initialize_from_actor(test_model, actor)
    unit_tests = run_unit_tests(test_model, inputs, device)
    del test_model
    if not all(unit_tests.values()):
        raise AssertionError(f"unit tests failed: {unit_tests}")
    print(json.dumps({"unit_tests": unit_tests}), flush=True)

    records: list[dict[str, Any]] = []
    for arm in arms:
        for fold in range(args.folds):
            heldout = set(fold_states[fold])
            train_states, val_states = inner_split[fold]
            heldout_frames = np.sort(np.concatenate([
                hard_states[key]["frames"] for key in fold_states[fold]
            ]))
            excluded = sorted(heldout | set(val_states) | control_states)
            easy_mask = ~hard_mask & ~np.isin(fresh_state, excluded)
            easy_pool = make_pool(
                fresh_context[easy_mask], fresh["actions"][easy_mask],
                fresh["transformed_reward"][easy_mask],
                args.primary_radius_index,
            )
            pool_by_state = {}
            for key in train_states:
                frames = hard_states[key]["frames"]
                pool_by_state[key] = make_pool(
                    fresh_context[frames], fresh["actions"][frames],
                    fresh["transformed_reward"][frames],
                    args.primary_radius_index,
                )
            states_by_stratum = {}
            for key in train_states:
                states_by_stratum.setdefault(
                    hard_states[key]["stratum_id"], []
                ).append(key)
            hard_sampler = {
                "strata": sorted(states_by_stratum),
                "states_by_stratum": states_by_stratum,
                "pool_by_state": pool_by_state,
                # Pooled std over all train-state deltas; a median of
                # per-state stds under-scales large-delta states and lets
                # them dominate the shared encoder.
                "delta_scale": float(np.std(np.concatenate([
                    pool["true_delta"] for pool in pool_by_state.values()
                ]))) + 1e-6,
            }
            if not val_states or not train_states:
                raise ValueError(
                    "fold split left no trainable or validation hard states"
                )
            hard_val_frames = np.sort(np.concatenate([
                hard_states[key]["frames"] for key in val_states
            ]))
            for seed in seeds:
                model, training = train_value_fold(
                    arm, fold, seed, actor, inputs, base_pool, easy_pool,
                    hard_sampler, validation_rows, hard_val_frames, fresh,
                    labels, args, device,
                )
                checkpoint_path = (
                    args.output_dir / f"vq2_{arm}_fold{fold}_seed{seed}.pt"
                )
                torch.save({
                    "format_version": 2,
                    "model_class": (
                        "TorchMPPISemanticInteractionStructuredLocalQCritic"
                        if arm == "PA_G0_X" else
                        "TorchMPPISemanticStructuredLocalQCritic"
                    ),
                    "prediction_path": "scalar value head; autograd gradient",
                    "model_state_dict": model.state_dict(),
                    "arm": arm, "fold": int(fold), "seed": int(seed),
                    "heldout_states": sorted(heldout),
                    "hard_validation_states": sorted(val_states),
                    "supervision": "relative probe value",
                    "initial_actor": str(args.initial_actor.resolve()),
                    "initial_actor_sha256": sha256_file(args.initial_actor),
                    "training": training,
                    "contract": {
                        "actor_frozen": True,
                        "formal_validation_loaded": False,
                        "test_loaded": False,
                        "unit_tests_passed": unit_tests,
                        "inner_split_shared_across_arms": True,
                        "easy_pool_excludes_inner_val_states": True,
                        "checkpoint_selected_by": "value-delta metrics",
                    },
                }, checkpoint_path)
                oof_value = value_metrics(
                    model, inputs, fresh_context[heldout_frames],
                    fresh["actions"][heldout_frames],
                    fresh["transformed_reward"][heldout_frames],
                    args.primary_radius_index, float(fresh["probe_radii_sigma"][args.primary_radius_index]),
                    args.evaluation_batch_size, device,
                )
                oof_value_consistency = {
                    str(radius_index): value_metrics(
                        model, inputs, fresh_context[heldout_frames],
                        fresh["actions"][heldout_frames],
                        fresh["transformed_reward"][heldout_frames],
                        radius_index,
                        float(fresh["probe_radii_sigma"][radius_index]),
                        args.evaluation_batch_size, device,
                    )
                    for radius_index in (
                        index for index in range(3)
                        if index != args.primary_radius_index
                    )
                }
                train_frames = np.sort(np.concatenate([
                    hard_states[key]["frames"] for key in train_states
                ]))
                seen_value = {
                    k: v for k, v in value_metrics(
                        model, inputs, fresh_context[train_frames],
                        fresh["actions"][train_frames],
                        fresh["transformed_reward"][train_frames],
                        args.primary_radius_index, float(
                            fresh["probe_radii_sigma"][args.primary_radius_index]
                        ),
                        args.evaluation_batch_size, device,
                    ).items() if not k.startswith("_")
                }
                seen_gradient = {
                    k: v for k, v in gradient_metrics(
                        model, inputs, fresh_context[train_frames],
                        fresh["actor_center"][train_frames],
                        fresh["gradient"][train_frames].astype(np.float32),
                        args.evaluation_batch_size, device,
                    ).items() if not k.startswith("_")
                }
                oof_gradient = gradient_metrics(
                    model, inputs, fresh_context[heldout_frames],
                    fresh["actor_center"][heldout_frames],
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
                    by_stratum[stratum] = {
                        "value": {
                            k: v for k, v in value_metrics(
                                model, inputs, fresh_context[stratum_frames],
                                fresh["actions"][stratum_frames],
                                fresh["transformed_reward"][stratum_frames],
                                args.primary_radius_index, float(fresh["probe_radii_sigma"][args.primary_radius_index]),
                                args.evaluation_batch_size, device,
                            ).items() if not k.startswith("_")
                        },
                        "gradient": {
                            k: v for k, v in gradient_metrics(
                                model, inputs, fresh_context[stratum_frames],
                                fresh["actor_center"][stratum_frames],
                                fresh["gradient"][stratum_frames].astype(np.float32),
                                args.evaluation_batch_size, device,
                            ).items() if not k.startswith("_")
                        },
                    }
                control_value = {
                    k: v for k, v in value_metrics(
                        model, inputs, fresh_context[strict_control_frames],
                        fresh["actions"][strict_control_frames],
                        fresh["transformed_reward"][strict_control_frames],
                        args.primary_radius_index, float(fresh["probe_radii_sigma"][args.primary_radius_index]),
                        args.evaluation_batch_size, device,
                    ).items() if not k.startswith("_")
                }
                control_gradient = {
                    k: v for k, v in gradient_metrics(
                        model, inputs, fresh_context[strict_control_frames],
                        fresh["actor_center"][strict_control_frames],
                        fresh["gradient"][strict_control_frames].astype(np.float32),
                        args.evaluation_batch_size, device,
                    ).items() if not k.startswith("_")
                }
                fresh230 = {
                    k: v for k, v in gradient_metrics(
                        model, inputs, fresh_unseen["context_index"],
                        fresh_unseen["actor_center"],
                        fresh_unseen["gradient"].astype(np.float32),
                        args.evaluation_batch_size, device,
                    ).items() if not k.startswith("_")
                }
                baseline = baseline_by_key.get((arm, fold, seed))
                records.append({
                    "arm": arm, "fold": int(fold), "seed": int(seed),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "heldout_states": sorted(heldout),
                    "heldout_frame_count": int(len(heldout_frames)),
                    "training": {
                        key: training[key] for key in (
                            "arm", "fold", "seed", "best_epoch", "epochs_run",
                            "best_validation_score",
                        )
                    },
                    "seen_hard_train_value": seen_value,
                    "seen_hard_train_gradient": seen_gradient,
                    "oof_value_primary": {
                        k: v for k, v in oof_value.items()
                        if not k.startswith("_")
                    },
                    "oof_value_consistency": {
                        name: {k: v for k, v in metrics.items()
                               if not k.startswith("_")}
                        for name, metrics in oof_value_consistency.items()
                    },
                    "oof_gradient": {
                        k: v for k, v in oof_gradient.items()
                        if not k.startswith("_")
                    },
                    "oof_by_stratum": by_stratum,
                    "control_20_state_strict": {
                        "value": control_value, "gradient": control_gradient,
                    },
                    "fresh_fd_unseen_230": fresh230,
                    "value_gates": {
                        "correlation_ge_0_50": (
                            oof_value["delta_correlation"] >= 0.50
                        ),
                        "rmse_over_scale_le_0_80": (
                            oof_value["delta_rmse_over_scale"] <= 0.80
                        ),
                        "sign_ge_0_60": (
                            oof_value["pair_sign_accuracy"] >= 0.60
                        ),
                    },
                    "derived_gradient_gate": {
                        "cosine_median_ge_0": (
                            oof_gradient["cosine_median"] >= 0.0
                        ),
                    },
                    "paired_vs_gradient_supervision": (
                        {
                            "cosine_median_delta": (
                                oof_gradient["cosine_median"]
                                - baseline["out_of_fold"]["cosine_median"]
                            ),
                            "cosine_p10_delta": (
                                oof_gradient["cosine_p10"]
                                - baseline["out_of_fold"]["cosine_p10"]
                            ),
                        }
                        if baseline else None
                    ),
                    "_oof_gradient_cosine": oof_gradient["_cosine"].tolist(),
                    "_oof_value_predicted": oof_value["_predicted"].tolist(),
                    "_oof_value_true": oof_value["_true"].tolist(),
                })
                del model

    # Strict pooled per seed: concatenate every seed's OOF frames.
    pooled: dict[str, Any] = {}
    for arm in arms:
        pooled[arm] = {}
        for seed in seeds:
            runs = [
                r for r in records
                if r["arm"] == arm and r["seed"] == seed
            ]
            cosine = np.concatenate([
                np.asarray(r["_oof_gradient_cosine"]) for r in runs
            ])
            predicted = np.concatenate([
                np.asarray(r["_oof_value_predicted"]).reshape(-1) for r in runs
            ])
            true = np.concatenate([
                np.asarray(r["_oof_value_true"]).reshape(-1) for r in runs
            ])
            pooled[arm][str(seed)] = {
                "gradient_cosine_median": float(np.median(cosine)),
                "gradient_cosine_p10": float(np.quantile(cosine, 0.10)),
                "value_delta_correlation": float(np.corrcoef(
                    predicted, true
                )[0, 1]),
                "value_delta_rmse_over_scale": float(
                    np.sqrt(np.mean(np.square(predicted - true)))
                    / (float(np.std(true)) + 1e-6)
                ),
                "value_pair_sign_accuracy": float(np.mean(
                    np.sign(predicted) == np.sign(true)
                )),
            }
    for record in records:
        for key in ("_oof_gradient_cosine", "_oof_value_predicted",
                    "_oof_value_true"):
            record.pop(key, None)

    decision = {"arms": {}}
    for arm in arms:
        arm_records = [r for r in records if r["arm"] == arm]
        value_pass = sum(
            all(r["value_gates"].values())
            and r["derived_gradient_gate"]["cosine_median_ge_0"]
            for r in arm_records
        )
        fold_pass = {}
        for fold in range(args.folds):
            runs = [r for r in arm_records if r["fold"] == fold]
            passing = sum(
                all(r["value_gates"].values())
                and r["derived_gradient_gate"]["cosine_median_ge_0"]
                for r in runs
            )
            fold_pass[str(fold)] = {
                "seed_pass_count": int(passing),
                "fold_passes": passing >= 2,
            }
        seed_pass = {}
        for seed in seeds:
            runs = [r for r in arm_records if r["seed"] == seed]
            seed_pass[str(seed)] = {
                "full_pass": bool(all(
                    all(r["value_gates"].values())
                    and r["derived_gradient_gate"]["cosine_median_ge_0"]
                    for r in runs
                )),
            }
        decision["arms"][arm] = {
            "runs_passing_all_value_and_gradient_gates": int(value_pass),
            "fold_pass": fold_pass,
            "folds_passing": int(sum(
                entry["fold_passes"] for entry in fold_pass.values()
            )),
            "seed_full_pass": seed_pass,
            "seeds_passing": int(sum(
                entry["full_pass"] for entry in seed_pass.values()
            )),
            "pooled_per_seed": pooled[arm],
            "paired_vs_gradient_median": float(np.median([
                r["paired_vs_gradient_supervision"]["cosine_median_delta"]
                for r in arm_records
            ])),
        }
    any_pass = any(
        entry["folds_passing"] >= 4 and entry["seeds_passing"] >= 2
        for entry in decision["arms"].values()
    )
    qualification = (
        "G0_VALUE_DELTA_V2_PASS_TARGET_REFORMULATION_CONFIRMED"
        if any_pass else "G0_VALUE_DELTA_V2_FAIL_COVERAGE_OR_CONTEXT_SUSPECTED"
    )
    summary = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "unit_tests": unit_tests,
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "labels": str(args.labels_npz.resolve()),
            "labels_sha256": sha256_file(args.labels_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "gradient_baseline": str(args.gradient_baseline.resolve()),
            "gradient_baseline_sha256": sha256_file(args.gradient_baseline),
        },
        "protocol": {
            "arms": arms, "folds": args.folds, "seeds": seeds,
            "prediction_path": (
                "scalar Q(s,a)=value_head(trunk(encoder(s,a))); "
                "gradient via autograd dQ/da at a0; g0/H heads dead"
            ),
            "supervision": "relative probe value Delta Q",
            "primary_radius_sigma": float(
                fresh["probe_radii_sigma"][args.primary_radius_index]
            ),
            "checkpoint_selection": "value-delta metrics only",
            "inner_split": "fixed per fold, shared across arms and seeds",
            "value_gates": {
                "correlation": 0.50, "rmse_over_scale": 0.80,
                "sign": 0.60, "derived_gradient_cosine_median": 0.0,
                "fold_rule": ">=2/3 seeds; >=4/5 folds",
                "seed_rule": ">=2/3 seeds full pass",
            },
        },
        "fold_assignment": assignment,
        "records": records,
        "decision": decision,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "supersedes_invalidated_v1": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": qualification,
        "unit_tests": unit_tests,
        "arms": {
            arm: {
                "runs_passing": d["runs_passing_all_value_and_gradient_gates"],
                "folds_passing": d["folds_passing"],
                "seeds_passing": d["seeds_passing"],
                "pooled_seed0": d["pooled_per_seed"]["0"],
            }
            for arm, d in decision["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
