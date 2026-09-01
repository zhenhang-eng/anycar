#!/usr/bin/env python3
"""Final paired Critic coordinate-conditioning experiment.

This is a deliberately narrow last-chance test for the scalar-Q Critic.  It
reuses the corrected value-delta v2 training pipeline without changing its
loss, optimizer, data, checkpoint selection, or state encoder.  The only
training difference is the action coordinate supplied to the scalar Q:

  A: Q(s, a), the existing physical 8x2 knot coordinate.
  Z: Q(s, z), with a = B z and a fold-train-only, full-rank B.

B combines the fixed per-channel MPPI sigma, a full-rank temporal DCT, and a
clipped diagonal scale fitted from the existing per-step position response on
the training states of each fold.  No heldout state is used to fit B.  A
coordinate-only A->Z metric is recomputed from the A checkpoint after training
to separate metric changes from genuine learning changes.

Every gradient is also mapped back to a physical action update and evaluated
with deterministic DBM J50 line searches at matched RMS sigma radii.  The
formal validation and test splits remain sealed, and the deployed Actor stays
frozen.
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

import run_mppi_g0_value_delta_cv as base
from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_g0_grouped_cv import HARD_STRATA
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_structured_local_q_critic import load_npz, set_seed


DEFAULT_POSITION = Path(
    "outputs/mppi_proposal/position_gradient_horizon_20260817_v1/"
    "per_step_position_gradients.npz"
)
DEFAULT_POSITION_ANALYSIS = Path(
    "outputs/mppi_proposal/position_gradient_horizon_20260817_v1/analysis.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/g0_sensitivity_coordinate_cv_20260818_v1"
)
DEFAULT_DERIVED_BASELINE = Path(
    "outputs/mppi_proposal/g0_sensitivity_coordinate_cv_20260818_v1_baseline.json"
)
MODES = {"PA_G0": "A", "PA_G0_COORD": "Z"}
RADII = (0.02, 0.05, 0.10)
EARLY_STEERING = np.asarray((1, 3, 5), np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=base.DEFAULT_INITIAL)
    parser.add_argument("--labels-npz", type=Path, default=base.DEFAULT_LABELS)
    parser.add_argument("--fresh-npz", type=Path, default=base.DEFAULT_FRESH)
    parser.add_argument(
        "--targeted-labels-npz", type=Path,
        default=base.DEFAULT_TARGETED_LABELS,
    )
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument(
        "--gradient-baseline", type=Path,
        default=base.DEFAULT_GRADIENT_BASELINE,
    )
    parser.add_argument("--position-npz", type=Path, default=DEFAULT_POSITION)
    parser.add_argument(
        "--position-analysis", type=Path, default=DEFAULT_POSITION_ANALYSIS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--derived-baseline", type=Path, default=DEFAULT_DERIVED_BASELINE,
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--selection-min-epoch", type=int, default=80)
    parser.add_argument("--scheduler-min-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batches-per-pool", type=int, default=8)
    parser.add_argument("--fold-seed", type=int, default=260814)
    parser.add_argument("--scale-clip-min", type=float, default=0.25)
    parser.add_argument("--scale-clip-max", type=float, default=4.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def dct_matrix(size: int) -> np.ndarray:
    result = np.empty((size, size), np.float64)
    for time in range(size):
        for mode in range(size):
            factor = np.sqrt(1.0 / size) if mode == 0 else np.sqrt(2.0 / size)
            result[time, mode] = factor * np.cos(
                np.pi * (time + 0.5) * mode / size
            )
    if not np.allclose(result.T @ result, np.eye(size), atol=1e-12):
        raise AssertionError("DCT basis is not orthonormal")
    return result


def temporal_action_basis() -> np.ndarray:
    temporal = dct_matrix(8)
    result = np.zeros((16, 16), np.float64)
    for knot in range(8):
        for mode in range(8):
            for channel in range(2):
                result[2 * knot + channel, 2 * mode + channel] = temporal[knot, mode]
    return result


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


class ExperimentContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.mode = "A"
        self.fold = 0
        self.transforms: dict[int, dict[str, np.ndarray | float | list[float]]] = {}
        self.a_state_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.reversal_cache: dict[tuple[int, str, int], dict[str, Any]] = {}

        self.fresh = load_npz(args.fresh_npz)
        self.manifest = json.loads(args.manifest.read_text())
        self.targeted = load_npz(args.targeted_labels_npz)
        self.fresh_context = self.fresh["context_index"].astype(np.int64)
        row_by_context = {
            int(row["context_index"]): row for row in self.manifest["rows"]
        }
        self.row_by_context = row_by_context
        self.fresh_state = np.asarray([
            f"{row_by_context[int(value)]['episode']}#"
            f"{int(row_by_context[int(value)]['physical_snapshot_ordinal'])}"
            for value in self.fresh_context
        ])
        self.fresh_stratum = np.asarray([
            row_by_context[int(value)]["stratum_id"] for value in self.fresh_context
        ])
        self.fresh_scenario = np.asarray([
            row_by_context[int(value)]["scenario"] for value in self.fresh_context
        ])
        self.hard_states, self.fold_states = self._fold_assignment()

        position_analysis = json.loads(args.position_analysis.read_text())
        expected_hash = position_analysis["sources"]["fresh_fd_sha256"]
        actual_hash = sha256_file(args.fresh_npz)
        if expected_hash != actual_hash:
            raise AssertionError("position-gradient/fresh-FD source hash mismatch")
        with np.load(args.position_npz, allow_pickle=False) as payload:
            self.per_step = payload["per_step"].astype(np.float64)
        if self.per_step.shape != (len(self.fresh_context), 50, 16):
            raise AssertionError("per-step sensitivity shape mismatch")
        self._fit_transforms()

        self.data, self.inputs, self.backend, self.weights = self._load_runtime()
        self.context_to_fresh = {
            int(value): position for position, value in enumerate(self.fresh_context)
        }

    def state_of(self, context_index: int) -> str:
        row = self.row_by_context[int(context_index)]
        return f"{row['episode']}#{int(row['physical_snapshot_ordinal'])}"

    def _fold_assignment(self):
        hard_mask = np.isin(self.fresh_stratum, HARD_STRATA)
        hard_states: dict[str, dict[str, Any]] = {}
        for position in np.flatnonzero(hard_mask):
            key = self.fresh_state[position]
            entry = hard_states.setdefault(key, {
                "state_key": key,
                "strata": set(),
                "scenario": self.fresh_scenario[position],
                "frames": [],
            })
            entry["strata"].add(self.fresh_stratum[position])
            entry["frames"].append(int(position))
        for entry in hard_states.values():
            entry["stratum_id"] = next(
                stratum for stratum in HARD_STRATA if stratum in entry["strata"]
            )
            entry["repeat_count"] = len(entry["frames"])

        rng = np.random.default_rng(self.args.fold_seed)
        keys = sorted(hard_states)
        keys = [keys[index] for index in rng.permutation(len(keys))]
        ordered = sorted(
            keys, key=lambda key: HARD_STRATA.index(hard_states[key]["stratum_id"])
        )
        assignment: dict[str, int] = {}
        for key in ordered:
            meta = hard_states[key]
            scores = []
            for fold in range(self.args.folds):
                members = [name for name, owner in assignment.items() if owner == fold]
                scores.append((
                    sum(hard_states[name]["stratum_id"] == meta["stratum_id"] for name in members),
                    len(members),
                    sum(hard_states[name]["scenario"] == meta["scenario"] for name in members),
                    sum(hard_states[name]["repeat_count"] == meta["repeat_count"] for name in members),
                ))
            assignment[key] = min(range(self.args.folds), key=lambda fold: scores[fold])
        fold_states = {
            fold: sorted(key for key, owner in assignment.items() if owner == fold)
            for fold in range(self.args.folds)
        }
        self.fold_assignment = assignment
        return hard_states, fold_states

    def _fit_transforms(self) -> None:
        rotation = temporal_action_basis()
        sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
        sigma_rotation = np.diag(sigma) @ rotation
        for fold, heldout_states in self.fold_states.items():
            train_frames = np.flatnonzero(~np.isin(self.fresh_state, heldout_states))
            coefficient_per_step = np.einsum(
                "nti,ij->ntj", self.per_step[train_frames], sigma_rotation
            )
            raw = np.median(
                np.sum(np.abs(coefficient_per_step), axis=1), axis=0
            )
            positive = raw[raw > 1e-12]
            if len(positive) != 16:
                raise AssertionError("coordinate sensitivity contains a zero mode")
            center = float(np.exp(np.mean(np.log(positive))))
            scale = np.clip(
                raw / center, self.args.scale_clip_min, self.args.scale_clip_max
            )
            matrix = sigma_rotation @ np.diag(1.0 / scale)
            inverse = np.linalg.inv(matrix)
            if not np.allclose(inverse @ matrix, np.eye(16), atol=1e-9):
                raise AssertionError("coordinate transform is not invertible")
            self.transforms[fold] = {
                "B": matrix.astype(np.float32),
                "B_inverse": inverse.astype(np.float32),
                "raw_sensitivity": raw.astype(np.float32),
                "normalized_scale": scale.astype(np.float32),
                "condition_number": float(np.linalg.cond(matrix)),
                "train_frame_count": int(len(train_frames)),
                "heldout_states": list(heldout_states),
            }

    def _load_runtime(self):
        initial_payload = torch.load(self.args.initial_actor, map_location="cpu")
        alpha_payload = torch.load(
            initial_payload["base_alpha_checkpoint"], map_location="cpu"
        )
        old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
        data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
        tensors = tensorize(data, self.device)
        extra = extra_tensors(data, self.device)
        alpha_policy = make_base_policy(alpha_payload, self.device)
        _, _, _, alpha_center = deterministic_outputs(
            alpha_policy, tensors, extra, np.arange(len(data.episodes)),
            float(initial_payload["base_move_threshold"]),
            self.args.evaluation_batch_size, self.device,
        )
        inputs = list(actor_inputs(tensors, alpha_center, self.device))
        labels = load_npz(self.args.labels_npz)
        actor_center_by_context = np.zeros((len(data.episodes), 8, 2), np.float32)
        assigned = np.zeros(len(data.episodes), bool)
        for position, value in enumerate(labels["context_index"].astype(np.int64)):
            actor_center_by_context[value] = labels["actor_center"][position]
            assigned[value] = True
        if not np.all(assigned):
            raise AssertionError("absolute action map is incomplete")
        inputs[3] = torch.from_numpy(actor_center_by_context).to(self.device)
        backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**data.dbm_params))
        weights = TorchMPPICostWeights(**data.cost_weights)
        return data, tuple(inputs), backend, weights

    @property
    def B(self) -> np.ndarray:
        return np.asarray(self.transforms[self.fold]["B"], np.float32)

    @property
    def B_inverse(self) -> np.ndarray:
        return np.asarray(self.transforms[self.fold]["B_inverse"], np.float32)

    def physical_to_coordinate(self, action: torch.Tensor) -> torch.Tensor:
        inverse = torch.as_tensor(self.B_inverse, device=action.device, dtype=action.dtype)
        flat = action.reshape(len(action), 16)
        return (flat @ inverse.T).reshape(-1, 8, 2)

    def target_gradient(self, physical: np.ndarray) -> np.ndarray:
        if self.mode == "A":
            return physical.astype(np.float32)
        return (physical.astype(np.float32) @ self.B).astype(np.float32)

    def mapped_action_gradient(self, predicted: np.ndarray) -> np.ndarray:
        if self.mode == "A":
            return predicted.astype(np.float32)
        return (predicted.astype(np.float32) @ self.B.T).astype(np.float32)

    def line_search(
        self,
        contexts: np.ndarray,
        references: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, Any]:
        physical_direction = self.mapped_action_gradient(predicted)
        sigma = np.tile(np.asarray((0.25, 0.35), np.float32), 8)
        standardized = physical_direction / sigma[None]
        norm = np.sqrt(np.mean(np.square(standardized), axis=1, keepdims=True))
        unit = standardized / np.maximum(norm, 1e-12)
        candidates = [references.astype(np.float32)]
        for radius in RADII:
            delta = radius * sigma[None] * unit
            candidates.append(np.clip(
                references.reshape(len(references), 16) - delta, -1.0, 1.0
            ).reshape(-1, 8, 2).astype(np.float32))
        knots = np.stack(candidates, axis=1)
        all_cost = []
        for start in range(0, len(contexts), self.args.evaluation_batch_size):
            stop = min(start + self.args.evaluation_batch_size, len(contexts))
            context = np.asarray(contexts[start:stop], np.int64)
            knot_tensor = torch.from_numpy(knots[start:stop]).to(self.device)
            actions = interpolate_knots(knot_tensor, self.backend.horizon)
            reference = torch.from_numpy(
                self.data.direct_reference[context].astype(np.float32)
            ).to(self.device)
            if reference.shape[1] == self.backend.horizon + 1:
                reference = reference[:, 1:]
            with torch.no_grad():
                cost = batched_cost(
                    self.backend, self.weights, actions,
                    torch.from_numpy(
                        self.data.initial_state_six[context].astype(np.float32)
                    ).to(self.device),
                    torch.from_numpy(
                        self.data.current_action[context].astype(np.float32)
                    ).to(self.device),
                    reference,
                )
            all_cost.append(cost.cpu().numpy())
        cost = np.concatenate(all_cost)
        result = {}
        for index, radius in enumerate(RADII, start=1):
            gain = cost[:, 0] - cost[:, index]
            result[f"r{radius:.2f}"] = {
                "gain_median": float(np.median(gain)),
                "gain_p05": float(np.quantile(gain, 0.05)),
                "gain_worst": float(np.min(gain)),
                "positive_fraction": float(np.mean(gain > 0)),
                "_gain": gain,
            }
        return result

    def reversal_metrics(self, model) -> dict[str, Any]:
        cache_key = (id(model), self.mode, self.fold)
        if cache_key in self.reversal_cache:
            return self.reversal_cache[cache_key]
        target_rows = np.flatnonzero(self.targeted["pilot_role"] == "target")
        heldout = set(self.fold_states[self.fold])
        selected = [
            int(row) for row in target_rows
            if self.state_of(int(self.targeted["context_index"][row])) in heldout
        ]
        if not selected:
            result = {"pair_count": 0, "true_reversal_count": 0, "recall": None}
            self.reversal_cache[cache_key] = result
            return result
        contexts = np.repeat(
            self.targeted["context_index"][selected].astype(np.int64), 77
        )
        actions = self.targeted["outer_actions"][selected].reshape(-1, 8, 2)
        predicted = []
        model.eval()
        for start in range(0, len(contexts), self.args.evaluation_batch_size):
            stop = min(start + self.args.evaluation_batch_size, len(contexts))
            predicted.append(q_gradient(
                model, self.inputs,
                torch.from_numpy(contexts[start:stop]).to(self.device),
                torch.from_numpy(actions[start:stop]).to(self.device),
            ).detach().cpu().numpy())
        predicted = self.mapped_action_gradient(np.concatenate(predicted)).reshape(
            len(selected), 77, 16
        )
        true = self.targeted["local_gradient"][selected].astype(np.float32)
        true_flags, predicted_flags = [], []
        for plus_start, minus_start in ((1, 20), (39, 58)):
            for direction in range(19):
                true_cos = cosine_rows(
                    true[:, plus_start + direction], true[:, minus_start + direction]
                )
                predicted_cos = cosine_rows(
                    predicted[:, plus_start + direction],
                    predicted[:, minus_start + direction],
                )
                true_flags.append(true_cos < -0.30)
                predicted_flags.append(predicted_cos < -0.30)
        true_flag = np.concatenate(true_flags)
        predicted_flag = np.concatenate(predicted_flags)
        count = int(np.sum(true_flag))
        result = {
            "pair_count": int(len(true_flag)),
            "true_reversal_count": count,
            "recall": float(np.mean(predicted_flag[true_flag])) if count else None,
            "predicted_reversal_fraction": float(np.mean(predicted_flag)),
        }
        self.reversal_cache[cache_key] = result
        return result


EXPERIMENT: ExperimentContext
ORIGINAL_BUILD_MODEL = base.build_model
ORIGINAL_TRAIN_VALUE_FOLD = base.train_value_fold
ORIGINAL_SCALAR_Q = base.scalar_q


def build_model(arm: str, args, device):
    return ORIGINAL_BUILD_MODEL("PA_G0", args, device)


def scalar_q(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    absolute_action: torch.Tensor,
) -> torch.Tensor:
    if EXPERIMENT.mode != "Z":
        return ORIGINAL_SCALAR_Q(model, inputs, context, absolute_action)
    coordinate = EXPERIMENT.physical_to_coordinate(absolute_action)
    feature = model.trunk(model.encoder(
        inputs[0][context], inputs[1][context], inputs[2][context],
        coordinate, inputs[4][context], inputs[5][context],
    ))
    return model.value_head(feature).squeeze(-1)


def probe_delta_value(
    model,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    reference_action: torch.Tensor,
    absolute_action: torch.Tensor,
) -> torch.Tensor:
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
    if EXPERIMENT.mode in ("A", "AT"):
        action = reference_action.detach().clone().requires_grad_(True)
        value = ORIGINAL_SCALAR_Q(model, inputs, context, action)
        physical = torch.autograd.grad(value.sum(), action)[0].reshape(len(context), 16)
        if EXPERIMENT.mode == "A":
            return physical
        matrix = torch.as_tensor(
            EXPERIMENT.B, device=physical.device, dtype=physical.dtype
        )
        return physical @ matrix
    coordinate = EXPERIMENT.physical_to_coordinate(reference_action).detach()
    coordinate.requires_grad_(True)
    feature = model.trunk(model.encoder(
        inputs[0][context], inputs[1][context], inputs[2][context],
        coordinate, inputs[4][context], inputs[5][context],
    ))
    value = model.value_head(feature).squeeze(-1)
    return torch.autograd.grad(value.sum(), coordinate)[0].reshape(len(context), 16)


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
        predicted.append(q_gradient(
            model, inputs,
            torch.from_numpy(np.asarray(contexts[start:stop])).to(device),
            torch.from_numpy(np.asarray(references[start:stop], np.float32)).to(device),
        ).detach().cpu().numpy())
    predicted = np.concatenate(predicted).astype(np.float32)
    target = EXPERIMENT.target_gradient(target_gradient)
    cosine = cosine_rows(predicted, target)
    ratio = np.linalg.norm(predicted, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    mapped = EXPERIMENT.mapped_action_gradient(predicted)
    action_cosine = cosine_rows(mapped, target_gradient.astype(np.float32))
    early_cosine = cosine_rows(
        mapped[:, EARLY_STEERING],
        target_gradient[:, EARLY_STEERING].astype(np.float32),
    )
    line = EXPERIMENT.line_search(contexts, references, predicted)
    reversal = EXPERIMENT.reversal_metrics(model)
    return {
        "count": int(len(contexts)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "norm_ratio_median": float(np.median(ratio)),
        "action_step_cosine_median": float(np.median(action_cosine)),
        "action_step_cosine_p10": float(np.quantile(action_cosine, 0.10)),
        "early_steering_cosine_median": float(np.median(early_cosine)),
        "early_steering_cosine_p10": float(np.quantile(early_cosine, 0.10)),
        "line_search": {
            key: clean_metrics(value) for key, value in line.items()
        },
        "reversal": reversal,
        "_cosine": cosine,
        "_action_cosine": action_cosine,
        "_early_cosine": early_cosine,
        "_line_gain": {key: value["_gain"] for key, value in line.items()},
    }


def train_value_fold(
    arm, fold, seed, actor, inputs,
    base_pool, easy_pool, hard_sampler,
    validation_rows, hard_val_frames, fresh, labels,
    args, device,
):
    EXPERIMENT.mode = MODES[arm]
    EXPERIMENT.fold = int(fold)
    model, training = ORIGINAL_TRAIN_VALUE_FOLD(
        "PA_G0", fold, seed, actor, inputs,
        base_pool, easy_pool, hard_sampler,
        validation_rows, hard_val_frames, fresh, labels,
        args, device,
    )
    training["arm"] = arm
    if EXPERIMENT.mode == "A":
        EXPERIMENT.a_state_cache[(int(fold), int(seed))] = copy.deepcopy(
            model.state_dict()
        )
    return model, training


def derived_baseline(source: Path, destination: Path, folds: int, seeds: list[int]) -> None:
    payload = json.loads(source.read_text())
    records = []
    existing = {
        (row["fold"], row["seed"]): row
        for row in payload["records"] if row["arm"] == "PA_G0"
    }
    for arm in MODES:
        for fold in range(folds):
            for seed in seeds:
                row = copy.deepcopy(existing[(fold, seed)])
                row["arm"] = arm
                records.append(row)
    destination.write_text(json.dumps({
        "format_version": 1,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "records": records,
    }, indent=2) + "\n")


def base_args(args: argparse.Namespace, seeds: list[int]) -> argparse.Namespace:
    return argparse.Namespace(
        initial_actor=args.initial_actor,
        labels_npz=args.labels_npz,
        fresh_npz=args.fresh_npz,
        targeted_labels_npz=args.targeted_labels_npz,
        manifest=args.manifest,
        gradient_baseline=args.derived_baseline,
        output_dir=args.output_dir,
        arms=",".join(MODES),
        folds=args.folds,
        seeds=",".join(map(str, seeds)),
        epochs=args.epochs,
        patience=args.patience,
        selection_min_epoch=args.selection_min_epoch,
        scheduler_min_epoch=args.scheduler_min_epoch,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=0.0,
        hessian_scale=256.0,
        small_chord_sigma=0.15,
        medium_chord_sigma=0.30,
        batches_per_pool=args.batches_per_pool,
        hard_validation_state_fraction=0.2,
        primary_radius_index=1,
        base_radius_index=0,
        fold_seed=args.fold_seed,
        device=args.device,
    )


def evaluate_checkpoint(record: dict[str, Any], mode: str) -> dict[str, Any]:
    EXPERIMENT.mode = mode
    EXPERIMENT.fold = int(record["fold"])
    model = build_model("PA_G0", BASE_ARGS, EXPERIMENT.device)
    payload = torch.load(record["checkpoint"], map_location=EXPERIMENT.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    heldout = set(record["heldout_states"])
    frames = np.flatnonzero(np.isin(EXPERIMENT.fresh_state, sorted(heldout)))
    metrics = gradient_metrics(
        model, EXPERIMENT.inputs,
        EXPERIMENT.fresh_context[frames],
        EXPERIMENT.fresh["actor_center"][frames],
        EXPERIMENT.fresh["gradient"][frames].astype(np.float32),
        BASE_ARGS.evaluation_batch_size, EXPERIMENT.device,
    )
    del model
    return metrics


def strict_pool(raw: list[dict[str, Any]]) -> dict[str, Any]:
    cosine = np.concatenate([entry["_cosine"] for entry in raw])
    action = np.concatenate([entry["_action_cosine"] for entry in raw])
    early = np.concatenate([entry["_early_cosine"] for entry in raw])
    line = {
        key: np.concatenate([entry["_line_gain"][key] for entry in raw])
        for key in raw[0]["_line_gain"]
    }
    return {
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "action_step_cosine_median": float(np.median(action)),
        "action_step_cosine_p10": float(np.quantile(action, 0.10)),
        "early_steering_cosine_median": float(np.median(early)),
        "early_steering_cosine_p10": float(np.quantile(early, 0.10)),
        "line_search": {
            key: {
                "gain_median": float(np.median(gain)),
                "gain_p05": float(np.quantile(gain, 0.05)),
                "gain_worst": float(np.min(gain)),
                "positive_fraction": float(np.mean(gain > 0)),
            }
            for key, gain in line.items()
        },
    }


def postprocess(args: argparse.Namespace, seeds: list[int]) -> None:
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    records = summary["records"]
    pooled: dict[str, dict[str, Any]] = {"A": {}, "AT": {}, "Z": {}}
    for mode, arm in (("A", "PA_G0"), ("Z", "PA_G0_COORD")):
        for seed in seeds:
            raw = []
            for record in records:
                if record["arm"] != arm or record["seed"] != seed:
                    continue
                metrics = evaluate_checkpoint(record, mode)
                record["final_action_coordinate_metrics"] = clean_metrics(metrics)
                raw.append(metrics)
            pooled[mode][str(seed)] = strict_pool(raw)

    for seed in seeds:
        raw = []
        for record in records:
            if record["arm"] != "PA_G0" or record["seed"] != seed:
                continue
            metrics = evaluate_checkpoint(record, "AT")
            record["coordinate_only_transformed_metrics"] = clean_metrics(metrics)
            raw.append(metrics)
        pooled["AT"][str(seed)] = strict_pool(raw)

    transforms = {}
    npz_payload = {}
    for fold, transform in EXPERIMENT.transforms.items():
        transforms[str(fold)] = {
            "raw_sensitivity": np.asarray(transform["raw_sensitivity"]).tolist(),
            "normalized_scale": np.asarray(transform["normalized_scale"]).tolist(),
            "condition_number": transform["condition_number"],
            "train_frame_count": transform["train_frame_count"],
            "heldout_states": transform["heldout_states"],
        }
        npz_payload[f"B_fold{fold}"] = transform["B"]
        npz_payload[f"B_inverse_fold{fold}"] = transform["B_inverse"]
    np.savez_compressed(args.output_dir / "coordinate_transforms.npz", **npz_payload)

    per_seed = {}
    for seed in seeds:
        a, at, z = pooled["A"][str(seed)], pooled["AT"][str(seed)], pooled["Z"][str(seed)]
        per_seed[str(seed)] = {
            "A": a, "A_transformed_only": at, "Z_retrained": z,
            "genuine_learning_delta_vs_transformed": {
                "cosine_median": z["cosine_median"] - at["cosine_median"],
                "cosine_p10": z["cosine_p10"] - at["cosine_p10"],
                "action_step_cosine_p10": (
                    z["action_step_cosine_p10"] - at["action_step_cosine_p10"]
                ),
            },
        }

    z_p10 = np.asarray([pooled["Z"][str(seed)]["cosine_p10"] for seed in seeds])
    z_median = np.asarray([pooled["Z"][str(seed)]["cosine_median"] for seed in seeds])
    z_action_p10 = np.asarray([
        pooled["Z"][str(seed)]["action_step_cosine_p10"] for seed in seeds
    ])
    line_p05 = np.asarray([
        pooled["Z"][str(seed)]["line_search"]["r0.05"]["gain_p05"]
        for seed in seeds
    ])
    genuine = np.asarray([
        per_seed[str(seed)]["genuine_learning_delta_vs_transformed"]["cosine_p10"]
        for seed in seeds
    ])
    actor_provider_pass = bool(
        np.sum((z_median >= 0.70) & (z_p10 >= 0.0)) >= 2
        and np.median(z_action_p10) >= 0.0
        and np.median(line_p05) >= 0.0
    )
    if actor_provider_pass:
        qualification = "COORDINATE_CRITIC_FULL_GATE_CANDIDATE_REOPEN_ONLY_AS_SECONDARY"
    elif np.median(z_p10) >= -0.20 and np.median(genuine) > 0:
        qualification = "COORDINATE_CRITIC_PARTIAL_RECOVERY_AUXILIARY_ONLY"
    else:
        qualification = "COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE"

    summary["format_version"] = 3
    summary["created_at_postprocess"] = datetime.now(timezone.utc).isoformat()
    summary["qualification"] = qualification
    # The reused value-delta runner still serializes its historical 5-fold
    # value gate.  This experiment runs three folds and does not use that gate
    # for the coordinate decision; make the distinction explicit in the
    # artifact instead of leaving an impossible "4/5 folds" rule behind.
    summary["protocol"]["value_gates"]["fold_rule"] = (
        "legacy value-fit diagnostic only; not used by this 3-fold "
        "coordinate experiment"
    )
    summary["protocol"]["value_gates"]["seed_rule"] = (
        "legacy value-fit diagnostic only; final decision uses strict pooled "
        "A/AT/Z metrics for all 3 seeds"
    )
    summary["coordinate_contract"] = {
        "physical_action_order": ["acceleration", "steering"],
        "early_steering_flat_indices": EARLY_STEERING.tolist(),
        "A": "Q(s,a), physical action coordinate",
        "AT": "same A checkpoint, metrics transformed by B^T only",
        "Z": "Q(s,z), a=Bz, retrained with unchanged value-delta loss",
        "B": "fold-train-only full-rank sigma x temporal-DCT x clipped sensitivity scale",
        "scale_clip": [args.scale_clip_min, args.scale_clip_max],
        "line_search_radii_sigma_rms": list(RADII),
        "formal_validation_loaded": False,
        "test_loaded": False,
        "actor_frozen": True,
    }
    summary["coordinate_transforms"] = transforms
    summary["strict_pooled_final"] = per_seed
    summary["final_decision"] = {
        "qualification": qualification,
        "median_z_cosine_p10": float(np.median(z_p10)),
        "median_genuine_p10_delta_vs_coordinate_only": float(np.median(genuine)),
        "median_z_action_step_p10": float(np.median(z_action_p10)),
        "median_z_r0p05_gain_p05": float(np.median(line_p05)),
        "actor_provider_full_gate_candidate": actor_provider_pass,
        "rule": (
            "P10<-0.2 or no genuine gain => close; -0.2<=P10<0 => auxiliary only; "
            "reopen only if >=2/3 seeds median>=0.70 and P10>=0 plus action-step/J50 tails"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["final_decision"], indent=2), flush=True)


def patch_base() -> None:
    base.build_model = build_model
    base.scalar_q = scalar_q
    base.probe_delta_value = probe_delta_value
    base.q_gradient = q_gradient
    base.gradient_metrics = gradient_metrics
    base.train_value_fold = train_value_fold
    base.parse_args = lambda: BASE_ARGS


BASE_ARGS: argparse.Namespace


def main() -> None:
    global EXPERIMENT, BASE_ARGS
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.derived_baseline.exists():
        raise FileExistsError(args.derived_baseline)
    derived_baseline(args.gradient_baseline, args.derived_baseline, args.folds, seeds)
    EXPERIMENT = ExperimentContext(args)
    BASE_ARGS = base_args(args, seeds)
    patch_base()
    try:
        base.main()
        postprocess(args, seeds)
    except Exception:
        raise


if __name__ == "__main__":
    main()
