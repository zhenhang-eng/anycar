#!/usr/bin/env python3
"""Validate direct risk objectives using frozen Query Decoder features.

The pointwise U2 head can consume the final state or all intermediate Decoder
states.  The S6-B head instead lets each output channel attend independently to
all Decoder layers and then applies lightweight causal temporal blocks across
the prediction horizon.  Both heads receive exactly the same explicit
conditions: future action, current context, nominal state, and nominal
transition.  They predict Gaussian sigma, expected absolute error, and the
probability of exceeding the train-set channel-wise 90th-percentile error
threshold while the deterministic mean model remains frozen.
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, str(REPO_ROOT / package_dir))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from car_foundation.kinematic_residual import KinematicBicycleParams
from car_foundation.probabilistic_residual import (
    gaussian_nll,
    horizon_channel_temperature,
    inverse_softplus,
    positive_scale,
)
from ablate_real_query_sigma_heads import (
    batch_from,
    cache_to,
    extract_cache as extract_final_cache,
    run_metrics,
    set_seed,
    split_files,
)
from ablate_real_query_uncertainty_structures import (
    extract_cache as extract_layer_cache,
)
from analyze_query_probability_shift import build_real_loader, load_query
from train_kinematic_residual_ablation import LOSS_WEIGHTS
from visualize_nuplan_query_probability_error import CHANNEL_NAMES, binary_auc


PROTOCOL = "real_query_s0_direct_risk_u2_v1"
SCORE_NAMES = ("s0_sigma", "u2_sigma", "u2_expected_abs", "u2_tail_probability")
LABELS = {
    "s0_sigma": "S0 Gaussian sigma",
    "u2_sigma": "U2 Gaussian sigma",
    "u2_expected_abs": "U2 predicted |error|",
    "u2_tail_probability": "U2 P(top-error)",
}
COLORS = {
    "s0_sigma": "#4C78A8",
    "u2_sigma": "#54A24B",
    "u2_expected_abs": "#F58518",
    "u2_tail_probability": "#E45756",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_real_finetune_query_baseline_split/20260728T143256/query_best.pt",
    )
    parser.add_argument(
        "--real-split-manifest-dir",
        type=Path,
        default=REPO_ROOT / "outputs/splits/session_isolated_real_v1",
    )
    parser.add_argument(
        "--s0-summary",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_real_query_uncertainty_s0_s3/20260728T162845/summary.json",
    )
    parser.add_argument(
        "--s0-data",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_real_query_uncertainty_s0_s3/20260728T162845/test_probability_error_data.npz",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--head-mode",
        choices=("pointwise", "layer_time"),
        default="pointwise",
        help=(
            "pointwise is the original shared U2 MLP; layer_time is S6-B with "
            "channel-specific layer attention and causal temporal blocks."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("final", "weighted", "concat"),
        default="final",
        help=(
            "final reproduces S0 features; weighted learns one global mixture "
            "of Decoder layers; concat preserves all three layer states."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--abs-floor", type=float, default=1e-6)
    parser.add_argument("--nll-weight", type=float, default=0.25)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/formal_real_query_direct_risk_u2",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size < 2 or args.feature_batch_size < 1:
        raise ValueError("batch sizes are invalid")
    if args.hidden_dim < 1 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("hidden dimension or dropout is invalid")
    if args.temporal_blocks < 1 or args.temporal_kernel_size < 2:
        raise ValueError("temporal block count and kernel size are invalid")
    if args.head_mode == "layer_time" and args.feature_mode != "concat":
        raise ValueError("layer_time requires --feature-mode concat")
    for name in (
        "nll_weight",
        "regression_weight",
        "classification_weight",
        "ranking_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def explicit_condition(batch):
    context = batch["context"][:, None, :].expand(-1, batch["action"].shape[1], -1)
    return torch.cat(
        (
            batch["action"],
            context,
            batch["nominal_state"],
            batch["nominal_transition"],
        ),
        dim=-1,
    )


class S0DirectRiskHead(nn.Module):
    def __init__(
        self,
        latent_dim,
        feature_mode,
        decoder_layers,
        condition_dim,
        hidden_dim,
        output_dim,
        dropout,
        sigma_floor,
        abs_floor,
    ):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.abs_floor = abs_floor
        self.feature_mode = feature_mode
        self.layer_logits = None
        feature_dim = latent_dim
        if feature_mode == "weighted":
            self.layer_logits = nn.Parameter(torch.zeros(decoder_layers))
        elif feature_mode == "concat":
            feature_dim = latent_dim * decoder_layers
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        self.abs_head = nn.Linear(hidden_dim, output_dim)
        self.tail_head = nn.Linear(hidden_dim, output_dim)

    def initialize_outputs(self, sigma, mean_abs, tail_rate):
        with torch.no_grad():
            for layer in (self.sigma_head, self.abs_head, self.tail_head):
                layer.weight.zero_()
            self.sigma_head.bias.copy_(inverse_softplus(sigma - self.sigma_floor))
            self.abs_head.bias.copy_(inverse_softplus(mean_abs - self.abs_floor))
            tail_rate = tail_rate.clamp(1e-4, 1.0 - 1e-4)
            self.tail_head.bias.copy_(torch.logit(tail_rate))

    def forward(self, batch):
        if self.feature_mode == "final":
            decoder_feature = batch["hidden"]
        elif self.feature_mode == "weighted":
            weights = torch.softmax(self.layer_logits, dim=0)
            decoder_feature = torch.sum(
                batch["hidden_layers"] * weights.view(1, 1, -1, 1), dim=2
            )
        else:
            decoder_feature = batch["hidden_layers"].flatten(start_dim=2)
        features = torch.cat((decoder_feature, explicit_condition(batch)), dim=-1)
        hidden = self.trunk(features)
        return {
            "sigma": positive_scale(self.sigma_head(hidden), self.sigma_floor),
            "expected_abs": positive_scale(self.abs_head(hidden), self.abs_floor),
            "tail_logit": self.tail_head(hidden),
        }


class ChannelProjection(nn.Module):
    """Project one risk token per physical output channel to a scalar."""

    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tokens):
        return torch.einsum("btch,ch->btc", tokens, self.weight) + self.bias


class CausalDepthwiseTemporalBlock(nn.Module):
    """Low-cost causal temporal mixing with a residual connection."""

    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        residual = tokens
        hidden = self.norm(tokens).transpose(1, 2)
        hidden = F.pad(hidden, (self.left_padding, 0))
        hidden = self.depthwise(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        return residual + self.dropout(F.silu(hidden))


class LayerTimeRiskHead(nn.Module):
    """S6-B: channel-specific layer fusion followed by causal time mixing."""

    def __init__(
        self,
        latent_dim,
        decoder_layers,
        condition_dim,
        hidden_dim,
        output_dim,
        prediction_length,
        dropout,
        temporal_blocks,
        temporal_kernel_size,
        sigma_floor,
        abs_floor,
    ):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.abs_floor = abs_floor
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.key_projection = nn.Linear(latent_dim, hidden_dim)
        self.value_projection = nn.Linear(latent_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.channel_queries = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.horizon_embedding = nn.Parameter(
            torch.empty(prediction_length, hidden_dim)
        )
        nn.init.normal_(self.channel_queries, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.layer_token_norm = nn.LayerNorm(hidden_dim)
        self.layer_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.temporal = nn.ModuleList(
            CausalDepthwiseTemporalBlock(
                hidden_dim,
                temporal_kernel_size,
                dilation=2**index,
                dropout=dropout,
            )
            for index in range(temporal_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.sigma_head = ChannelProjection(output_dim, hidden_dim)
        self.abs_head = ChannelProjection(output_dim, hidden_dim)
        self.tail_head = ChannelProjection(output_dim, hidden_dim)

    def initialize_outputs(self, sigma, mean_abs, tail_rate):
        with torch.no_grad():
            for layer in (self.sigma_head, self.abs_head, self.tail_head):
                layer.weight.zero_()
            self.sigma_head.bias.copy_(inverse_softplus(sigma - self.sigma_floor))
            self.abs_head.bias.copy_(inverse_softplus(mean_abs - self.abs_floor))
            tail_rate = tail_rate.clamp(1e-4, 1.0 - 1e-4)
            self.tail_head.bias.copy_(torch.logit(tail_rate))

    def _risk_tokens(self, batch):
        layers = self.layer_norm(batch["hidden_layers"])
        keys = self.key_projection(layers)
        values = self.value_projection(layers)
        condition = self.condition_projection(explicit_condition(batch))
        horizon = self.horizon_embedding[: layers.shape[1]].view(
            1, layers.shape[1], 1, self.hidden_dim
        )
        queries = (
            condition.unsqueeze(2)
            + self.channel_queries.view(1, 1, self.output_dim, self.hidden_dim)
            + horizon
        )
        attention_logits = torch.einsum("btch,btlh->btcl", queries, keys)
        attention = torch.softmax(attention_logits / math.sqrt(self.hidden_dim), dim=-1)
        attended = torch.einsum("btcl,btlh->btch", attention, values)
        tokens = self.layer_token_norm(attended + queries)
        tokens = tokens + self.layer_ffn(tokens)
        batch_size, horizon_length, channels, hidden_dim = tokens.shape
        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * channels, horizon_length, hidden_dim
        )
        for block in self.temporal:
            temporal = block(temporal)
        tokens = temporal.reshape(
            batch_size, channels, horizon_length, hidden_dim
        ).permute(0, 2, 1, 3)
        return self.output_norm(tokens), attention

    def forward(self, batch, return_attention=False):
        tokens, attention = self._risk_tokens(batch)
        output = {
            "sigma": positive_scale(self.sigma_head(tokens), self.sigma_floor),
            "expected_abs": positive_scale(self.abs_head(tokens), self.abs_floor),
            "tail_logit": self.tail_head(tokens),
        }
        if return_attention:
            output["layer_attention"] = attention
        return output


def training_targets(error, abs_scale, tail_threshold):
    absolute = error.abs()
    scaled_abs = absolute / abs_scale
    tail = (absolute >= tail_threshold).to(error.dtype)
    return absolute, scaled_abs, tail


def pairwise_rank_loss(predicted_abs, target_abs, abs_scale):
    shift = max(predicted_abs.shape[0] // 2, 1)
    paired_prediction = torch.roll(predicted_abs, shifts=shift, dims=0)
    paired_target = torch.roll(target_abs, shifts=shift, dims=0)
    target_delta = (target_abs - paired_target) / abs_scale
    prediction_delta = (predicted_abs - paired_prediction) / abs_scale
    sign = torch.sign(target_delta)
    valid = sign != 0
    if not torch.any(valid):
        return predicted_abs.sum() * 0.0
    # Large, unambiguous target differences contribute more, with clipping to
    # prevent rare residual outliers from dominating the rank objective.
    importance = torch.clamp(target_delta.abs(), max=2.0).detach()
    losses = F.softplus(-sign * prediction_delta) * importance
    return losses[valid].sum() / importance[valid].sum().clamp_min(1e-6)


def loss_components(head, batch, abs_scale, tail_threshold, weights, args):
    output = head(batch)
    absolute, scaled_abs, tail = training_targets(
        batch["error"], abs_scale, tail_threshold
    )
    nll = gaussian_nll(batch["error"], output["sigma"], weights=weights)
    regression = F.smooth_l1_loss(output["expected_abs"] / abs_scale, scaled_abs)
    classification = F.binary_cross_entropy_with_logits(output["tail_logit"], tail)
    ranking = pairwise_rank_loss(output["expected_abs"], absolute, abs_scale)
    total = (
        args.nll_weight * nll
        + args.regression_weight * regression
        + args.classification_weight * classification
        + args.ranking_weight * ranking
    )
    return total, {
        "nll": nll,
        "regression": regression,
        "classification": classification,
        "ranking": ranking,
    }


def evaluate_loss(head, cache, batch_size, abs_scale, tail_threshold, weights, args):
    totals = {key: 0.0 for key in ("total", "nll", "regression", "classification", "ranking")}
    count = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            batch = batch_from(cache, slice(start, start + batch_size))
            total, parts = loss_components(
                head, batch, abs_scale, tail_threshold, weights, args
            )
            examples = batch["error"].shape[0]
            totals["total"] += float(total.item()) * examples
            for key, value in parts.items():
                totals[key] += float(value.item()) * examples
            count += examples
    return {key: value / max(count, 1) for key, value in totals.items()}


def fit_head(train, val, args, device):
    set_seed(args.seed)
    abs_scale = train["error"].abs().mean(dim=0).clamp_min(args.abs_floor * 10.0)
    tail_threshold = torch.quantile(
        train["error"].abs().flatten(0, 1), 0.90, dim=0
    ).view(1, 1, -1)
    initial_sigma = torch.sqrt(train["error"].square().mean(dim=(0, 1))).clamp_min(
        args.sigma_floor * 1.01
    )
    initial_abs = train["error"].abs().mean(dim=(0, 1)).clamp_min(
        args.abs_floor * 1.01
    )
    initial_tail = (
        train["error"].abs() >= tail_threshold
    ).to(torch.float32).mean(dim=(0, 1))
    if args.head_mode == "layer_time":
        head = LayerTimeRiskHead(
            latent_dim=train["hidden_layers"].shape[-1],
            decoder_layers=train["hidden_layers"].shape[2],
            condition_dim=15,
            hidden_dim=args.hidden_dim,
            output_dim=train["error"].shape[-1],
            prediction_length=train["error"].shape[1],
            dropout=args.dropout,
            temporal_blocks=args.temporal_blocks,
            temporal_kernel_size=args.temporal_kernel_size,
            sigma_floor=args.sigma_floor,
            abs_floor=args.abs_floor,
        ).to(device)
    else:
        head = S0DirectRiskHead(
            latent_dim=(
                train["hidden"].shape[-1]
                if args.feature_mode == "final"
                else train["hidden_layers"].shape[-1]
            ),
            feature_mode=args.feature_mode,
            decoder_layers=(
                1
                if args.feature_mode == "final"
                else train["hidden_layers"].shape[2]
            ),
            condition_dim=15,
            hidden_dim=args.hidden_dim,
            output_dim=train["error"].shape[-1],
            dropout=args.dropout,
            sigma_floor=args.sigma_floor,
            abs_floor=args.abs_floor,
        ).to(device)
    head.initialize_outputs(initial_sigma, initial_abs, initial_tail)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    weights = LOSS_WEIGHTS.to(device)
    initial_val = evaluate_loss(
        head, val, args.batch_size, abs_scale, tail_threshold, weights, args
    )
    best_value = initial_val["total"]
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    history = [{"epoch": 0, "learning_rate": args.lr, **{f"val_{k}": v for k, v in initial_val.items()}}]
    stale = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(train["error"].shape[0], generator=generator)
        accumulated = {key: 0.0 for key in ("total", "nll", "regression", "classification", "ranking")}
        count = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            batch = batch_from(train, indices)
            total, parts = loss_components(
                head, batch, abs_scale, tail_threshold, weights, args
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            examples = batch["error"].shape[0]
            accumulated["total"] += float(total.detach().item()) * examples
            for key, value in parts.items():
                accumulated[key] += float(value.detach().item()) * examples
            count += examples
        train_values = {key: value / count for key, value in accumulated.items()}
        val_values = evaluate_loss(
            head, val, args.batch_size, abs_scale, tail_threshold, weights, args
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_values.items()},
                **{f"val_{k}": v for k, v in val_values.items()},
            }
        )
        print(
            f"u2 epoch={epoch:03d} train={train_values['total']:.6f} "
            f"val={val_values['total']:.6f} "
            f"reg={val_values['regression']:.5f} "
            f"cls={val_values['classification']:.5f} "
            f"rank={val_values['ranking']:.5f}",
            flush=True,
        )
        if val_values["total"] < best_value:
            best_value = val_values["total"]
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
        scheduler.step()
    head.load_state_dict(best_state)
    return head, {
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_val_total": best_value,
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "history": history,
        "abs_scale": abs_scale.detach().cpu(),
        "tail_threshold": tail_threshold.detach().cpu(),
        "state_dict": best_state,
    }


def collect_outputs(head, cache, batch_size):
    collected = {key: [] for key in ("sigma", "expected_abs", "tail_logit")}
    head.eval()
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            output = head(batch_from(cache, slice(start, start + batch_size)))
            for key in collected:
                collected[key].append(output[key].detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def collect_layer_attention(head, cache, batch_size):
    if not isinstance(head, LayerTimeRiskHead):
        return None
    collected = []
    head.eval()
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            output = head(
                batch_from(cache, slice(start, start + batch_size)),
                return_attention=True,
            )
            collected.append(output["layer_attention"].detach().cpu())
    return torch.cat(collected, dim=0)


def fit_calibration(head, calibration_gpu, calibration_cpu, fit, args):
    output = collect_outputs(head, calibration_gpu, args.batch_size)
    error = calibration_cpu["error"].abs()
    sigma_temperature = horizon_channel_temperature(
        calibration_cpu["error"], output["sigma"]
    )
    direct_scale = (
        error.mean(dim=0) / output["expected_abs"].mean(dim=0).clamp_min(args.abs_floor)
    ).clamp(0.1, 10.0)
    threshold = fit["tail_threshold"].view(1, 1, -1)
    target = (error >= threshold).to(torch.float32)
    platt_scale = []
    platt_bias = []
    for channel in range(error.shape[-1]):
        logits = output["tail_logit"][..., channel].reshape(-1)
        labels = target[..., channel].reshape(-1)
        raw_scale = nn.Parameter(inverse_softplus(torch.ones(())))
        bias = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.LBFGS(
            (raw_scale, bias), lr=0.5, max_iter=50, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            scale = F.softplus(raw_scale) + 1e-4
            loss = F.binary_cross_entropy_with_logits(logits * scale + bias, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        platt_scale.append(float((F.softplus(raw_scale) + 1e-4).detach()))
        platt_bias.append(float(bias.detach()))
    return {
        "sigma_temperature": sigma_temperature,
        "direct_scale": direct_scale,
        "platt_scale": torch.tensor(platt_scale),
        "platt_bias": torch.tensor(platt_bias),
    }


def rank_metrics(error, score):
    error_np = np.asarray(error, dtype=np.float64)
    score_np = np.asarray(score, dtype=np.float64)
    channels = {}
    for index, channel in enumerate(CHANNEL_NAMES):
        absolute = np.abs(error_np[..., index]).reshape(-1)
        risk = score_np[..., index].reshape(-1)
        high_error = absolute >= np.quantile(absolute, 0.90)
        high_score = risk >= np.quantile(risk, 0.90)
        channels[channel] = {
            "spearman_score_abs_error": float(spearmanr(risk, absolute).statistic),
            "top10_error_auc": float(binary_auc(risk, high_error)),
            "top10_error_recall_by_top10_score": float(
                np.sum(high_error & high_score) / np.sum(high_error)
            ),
        }
    aggregate = {
        key: float(np.nanmean([values[key] for values in channels.values()]))
        for key in (
            "spearman_score_abs_error",
            "top10_error_auc",
            "top10_error_recall_by_top10_score",
        )
    }
    return {"aggregate": aggregate, "channels": channels}


def binary_calibration_metrics(error, probability, threshold, bins=10):
    result = {}
    absolute = np.abs(error)
    for index, channel in enumerate(CHANNEL_NAMES):
        target = (absolute[..., index] >= threshold[index]).astype(np.float64).reshape(-1)
        score = probability[..., index].reshape(-1)
        groups = np.array_split(np.argsort(score), bins)
        predicted = np.asarray([score[group].mean() for group in groups])
        observed = np.asarray([target[group].mean() for group in groups])
        result[channel] = {
            "fixed_threshold": float(threshold[index]),
            "positive_rate": float(target.mean()),
            "brier": float(np.mean(np.square(score - target))),
            "equal_count_ece": float(np.mean(np.abs(predicted - observed))),
            "bin_predicted": predicted.tolist(),
            "bin_observed": observed.tolist(),
        }
    return result


def benchmark_head(head, cache, args, device):
    result = {}
    head.eval()
    with torch.no_grad():
        for requested in (1, 256):
            size = min(requested, cache["error"].shape[0])
            batch = batch_from(cache, slice(0, size))
            for _ in range(args.benchmark_warmup):
                head(batch)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.benchmark_iterations):
                head(batch)
            end.record()
            torch.cuda.synchronize(device)
            result[str(requested)] = start.elapsed_time(end) / args.benchmark_iterations
    return result


def plot_training(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train = [entry for entry in history if "train_total" in entry]
    axes[0].plot([e["epoch"] for e in train], [e["train_total"] for e in train], label="train")
    axes[0].plot([e["epoch"] for e in history], [e["val_total"] for e in history], label="validation")
    for key in ("regression", "classification", "ranking"):
        axes[1].plot([e["epoch"] for e in history], [e[f"val_{key}"] for e in history], label=key)
    axes[0].set(title="U2 weighted total objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation direct-risk objectives", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "01_training_objectives.png", dpi=180)
    plt.close(fig)


def plot_aggregate(metrics, output):
    specs = (
        ("spearman_score_abs_error", "Mean Spearman"),
        ("top10_error_auc", "Mean top-10% AUC"),
        ("top10_error_recall_by_top10_score", "Mean top-10% recall"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (key, title) in zip(axes, specs):
        values = [metrics[name]["aggregate"][key] for name in SCORE_NAMES]
        axis.bar(range(len(SCORE_NAMES)), values, color=[COLORS[name] for name in SCORE_NAMES])
        axis.set_xticks(range(len(SCORE_NAMES)), [name.replace("u2_", "U2\n").replace("s0_", "S0\n") for name in SCORE_NAMES], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "02_aggregate_risk_metrics.png", dpi=180)
    plt.close(fig)


def plot_channels(metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(CHANNEL_NAMES))
    width = 0.2
    for index, name in enumerate(SCORE_NAMES):
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, [metrics[name]["channels"][c]["spearman_score_abs_error"] for c in CHANNEL_NAMES], width, color=COLORS[name], label=LABELS[name])
        axes[1].bar(x + offset, [metrics[name]["channels"][c]["top10_error_auc"] for c in CHANNEL_NAMES], width, color=COLORS[name], label=LABELS[name])
    axes[0].set_title("Score-error Spearman")
    axes[1].set_title("Top-10% error AUC")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(x, CHANNEL_NAMES)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "03_channel_risk_metrics.png", dpi=180)
    plt.close(fig)


def plot_focus_metrics(metrics, output):
    focus = ("dy_body", "dvx", "dyawrate")
    specs = (
        ("spearman_score_abs_error", "Mean Spearman"),
        ("top10_error_auc", "Mean top-10% AUC"),
        ("top10_error_recall_by_top10_score", "Mean top-10% recall"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (key, title) in zip(axes, specs):
        values = [
            np.mean([metrics[name]["channels"][channel][key] for channel in focus])
            for name in SCORE_NAMES
        ]
        axis.bar(range(len(SCORE_NAMES)), values, color=[COLORS[name] for name in SCORE_NAMES])
        axis.set_xticks(
            range(len(SCORE_NAMES)),
            [name.replace("u2_", "U2\n").replace("s0_", "S0\n") for name in SCORE_NAMES],
            fontsize=8,
        )
        axis.set_title(f"{title}\n(excluding dx_body)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "06_focus_metrics_excluding_dx.png", dpi=180)
    plt.close(fig)


def plot_error_bins(error, scores, bins, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        absolute = np.abs(error[..., channel_index]).reshape(-1)
        for name in SCORE_NAMES:
            score = scores[name][..., channel_index].reshape(-1)
            groups = np.array_split(np.argsort(score), bins)
            observed = [absolute[group].mean() for group in groups]
            axis.plot(range(1, bins + 1), observed, marker="o", markersize=3, color=COLORS[name], label=LABELS[name])
        axis.set(title=channel, xlabel="Predicted-risk quantile bin")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed physical MAE")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "04_risk_bins_observed_error.png", dpi=180)
    plt.close(fig)


def plot_tail_reliability(calibration, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for axis, channel in zip(axes, CHANNEL_NAMES):
        values = calibration[channel]
        predicted = np.asarray(values["bin_predicted"])
        observed = np.asarray(values["bin_observed"])
        maximum = max(float(predicted.max()), float(observed.max()), 0.1)
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=1)
        axis.plot(predicted, observed, marker="o", color=COLORS["u2_tail_probability"])
        axis.set(title=channel, xlabel="Predicted tail probability")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed threshold exceedance rate")
    fig.tight_layout()
    fig.savefig(output / "05_tail_probability_reliability.png", dpi=180)
    plt.close(fig)


def plot_layer_attention(attention, output):
    if attention is None:
        return
    mean_attention = attention.mean(dim=0).numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    horizons = np.arange(1, mean_attention.shape[0] + 1)
    for channel_index, (axis, channel) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        for layer_index in range(mean_attention.shape[-1]):
            axis.plot(
                horizons,
                mean_attention[:, channel_index, layer_index],
                label=f"Decoder H{layer_index + 1}",
            )
        axis.set_title(channel)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Prediction horizon")
    axes[1, 1].set_xlabel("Prediction horizon")
    axes[0, 0].set_ylabel("Mean layer attention")
    axes[1, 0].set_ylabel("Mean layer attention")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("S6-B channel-specific Decoder-layer attention")
    fig.tight_layout()
    fig.savefig(output / "07_s6_layer_attention.png", dpi=180)
    plt.close(fig)


def normalized_selective_risk_curve(absolute_error, score, coverages):
    absolute_error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(score)
    cumulative_error = np.cumsum(absolute_error[order])
    counts = np.maximum(
        1, np.minimum(len(order), np.rint(coverages * len(order)).astype(int))
    )
    retained_mae = cumulative_error[counts - 1] / counts
    return retained_mae / max(float(absolute_error.mean()), 1e-12)


def plot_selective_risk(error, scores, output, label_prefix="U2"):
    coverages = np.linspace(0.10, 1.0, 46)
    selective_labels = {
        "s0_sigma": "S0 Gaussian sigma",
        "u2_sigma": f"{label_prefix} Gaussian sigma",
        "u2_expected_abs": f"{label_prefix} predicted |error|",
        "u2_tail_probability": f"{label_prefix} P(top-error)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    absolute = np.abs(error)
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        channel_error = absolute[..., channel_index]
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Random")
        axis.plot(
            coverages,
            normalized_selective_risk_curve(
                channel_error, channel_error, coverages
            ),
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Oracle",
        )
        for name in SCORE_NAMES:
            axis.plot(
                coverages,
                normalized_selective_risk_curve(
                channel_error, scores[name][..., channel_index], coverages
                ),
                color=COLORS[name],
                label=selective_labels[name],
            )
        axis.set_title(channel)
        axis.set_xlabel("Retained coverage")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Retained MAE / full-set MAE")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Selective risk: reject predictions with the highest predicted risk")
    fig.tight_layout()
    fig.savefig(output / "08_selective_risk_s6b.png", dpi=180)
    plt.close(fig)


def write_csv(metrics, output):
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("score", "scope", "channel", "metric", "value"))
        for name, result in metrics.items():
            for metric, value in result["aggregate"].items():
                writer.writerow((name, "aggregate", "mean", metric, value))
            for channel, values in result["channels"].items():
                for metric, value in values.items():
                    writer.writerow((name, "channel", channel, metric, value))


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device(args.device)
    output = args.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    splits = split_files(args.real_split_manifest_dir.resolve())
    for name, files in splits.items():
        (output / f"{name}_files.txt").write_text("\n".join(files) + "\n")

    checkpoint, model_args, model, stats = load_query(args.mean_checkpoint.resolve(), device)
    params = KinematicBicycleParams(**checkpoint["params"])
    dataset_args = SimpleNamespace(**vars(args))
    dataset_args.eval_batch_size = args.feature_batch_size
    datasets = {}
    audits = {}
    caches_cpu = {}
    hidden_audits = {}
    for name in ("sigma_train", "sigma_val", "calibration"):
        datasets[name], loader, audits[name] = build_real_loader(
            splits[name], model_args, params, dataset_args, name, shuffle=False
        )
        if args.feature_mode == "final":
            caches_cpu[name] = extract_final_cache(model, loader, stats, device, name)
        else:
            caches_cpu[name], hidden_audits[name] = extract_layer_cache(
                model, loader, stats, device, name
            )
            # U2 reads Decoder states only. History Memory is intentionally
            # removed so concat does not accidentally become an S5 experiment.
            caches_cpu[name].pop("history_memory")
    caches = {name: cache_to(cache, device) for name, cache in caches_cpu.items()}

    head, fit = fit_head(caches["sigma_train"], caches["sigma_val"], args, device)
    calibration = fit_calibration(
        head, caches["calibration"], caches_cpu["calibration"], fit, args
    )
    torch.save(
        {
            "protocol": f"{PROTOCOL}_{args.feature_mode}_{args.head_mode}",
            "state_dict": fit["state_dict"],
            "abs_scale": fit["abs_scale"],
            "tail_threshold": fit["tail_threshold"],
            "calibration": calibration,
            "mean_checkpoint": str(args.mean_checkpoint.resolve()),
            "args": vars(args),
        },
        output / f"u2_{args.feature_mode}_{args.head_mode}_direct_risk_best.pt",
    )

    # Test data are materialized only after checkpoint selection and all
    # calibration parameters have been fitted.
    datasets["real_test"], test_loader, audits["real_test"] = build_real_loader(
        splits["real_test"], model_args, params, dataset_args, "real_test", shuffle=False
    )
    if args.feature_mode == "final":
        test_cpu = extract_final_cache(model, test_loader, stats, device, "real_test")
    else:
        test_cpu, hidden_audits["real_test"] = extract_layer_cache(
            model, test_loader, stats, device, "real_test"
        )
        test_cpu.pop("history_memory")
    test = cache_to(test_cpu, device)
    raw = collect_outputs(head, test, args.batch_size)
    layer_attention = collect_layer_attention(head, test, args.batch_size)
    scale = stats["residual"][1].detach().cpu().view(1, 1, -1)
    physical_error = test_cpu["error"] * scale
    u2_sigma = raw["sigma"] * calibration["sigma_temperature"].view(1, *calibration["sigma_temperature"].shape)
    u2_sigma_physical = u2_sigma * scale
    u2_abs = raw["expected_abs"] * calibration["direct_scale"].view(1, *calibration["direct_scale"].shape)
    u2_abs_physical = u2_abs * scale
    tail_probability = torch.sigmoid(
        raw["tail_logit"] * calibration["platt_scale"].view(1, 1, -1)
        + calibration["platt_bias"].view(1, 1, -1)
    )

    base_summary = json.loads(args.s0_summary.read_text())
    base_data = np.load(args.s0_data)
    base_error = np.asarray(base_data["error"], dtype=np.float64)
    if args.max_episodes_per_split > 0:
        base_error = base_error[: physical_error.shape[0]]
    if args.max_episodes_per_split == 0 and not np.allclose(
        base_error, physical_error.numpy(), rtol=0.0, atol=1e-8
    ):
        raise RuntimeError("S0 reference and U2 test errors differ")
    scores = {
        "s0_sigma": np.asarray(
            base_data["sigma_s0_conditional_mlp"][: physical_error.shape[0]],
            dtype=np.float64,
        ),
        "u2_sigma": u2_sigma_physical.numpy().astype(np.float64),
        "u2_expected_abs": u2_abs_physical.numpy().astype(np.float64),
        "u2_tail_probability": tail_probability.numpy().astype(np.float64),
    }
    error_np = physical_error.numpy().astype(np.float64)
    metrics = {name: rank_metrics(error_np, value) for name, value in scores.items()}
    u2_sigma_metrics = run_metrics(physical_error, u2_sigma_physical, args.bins)
    test_nll = float(
        gaussian_nll(
            test["error"], u2_sigma.to(device), weights=LOSS_WEIGHTS.to(device), include_constant=True
        ).item()
    )
    physical_threshold = (
        fit["tail_threshold"].view(-1) * scale.view(-1)
    ).numpy()
    tail_calibration = binary_calibration_metrics(
        error_np, scores["u2_tail_probability"], physical_threshold
    )
    benchmark = benchmark_head(head, test, args, device)

    compact_fit = {
        "best_epoch": fit["best_epoch"],
        "stopped_epoch": fit["stopped_epoch"],
        "best_val_total": fit["best_val_total"],
        "parameter_count": fit["parameter_count"],
        "history": fit["history"],
        "abs_scale": fit["abs_scale"].tolist(),
        "tail_threshold": fit["tail_threshold"].tolist(),
        "benchmark_ms": benchmark,
    }
    plot_training(fit["history"], output)
    plot_aggregate(metrics, output)
    plot_channels(metrics, output)
    plot_error_bins(error_np, scores, args.bins, output)
    plot_tail_reliability(tail_calibration, output)
    plot_focus_metrics(metrics, output)
    plot_layer_attention(layer_attention, output)
    plot_selective_risk(
        error_np,
        scores,
        output,
        label_prefix="S6-B" if args.head_mode == "layer_time" else "U2",
    )
    write_csv(metrics, output)
    np.savez_compressed(
        output / "test_risk_outputs.npz",
        error=error_np,
        **scores,
        channel_names=np.asarray(CHANNEL_NAMES),
    )
    summary = {
        "protocol": f"{PROTOCOL}_{args.feature_mode}_{args.head_mode}",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "strict_controls": {
            "feature_mode": args.feature_mode,
            "head_mode": args.head_mode,
            "s0_final_decoder_hidden_only": args.feature_mode == "final",
            "all_decoder_layer_states_without_history_memory": args.feature_mode != "final",
            "channel_specific_layer_attention": args.head_mode == "layer_time",
            "causal_temporal_risk_blocks": (
                args.temporal_blocks if args.head_mode == "layer_time" else 0
            ),
            "same_explicit_conditions_as_s0": True,
            "mean_model_frozen": True,
            "thresholds_and_target_scales_from_sigma_train_only": True,
            "selection_from_sigma_validation_only": True,
            "calibration_from_calibration_only": True,
            "test_loaded_after_selection_and_calibration": True,
            "s0_reference_test_error_exact_match": True,
        },
        "mean_checkpoint": str(args.mean_checkpoint.resolve()),
        "mean_checkpoint_epoch": int(checkpoint["epoch"]),
        "split_episode_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "split_audits": audits,
        "final_hidden_equivalence_max_abs": hidden_audits,
        "fit": compact_fit,
        "rank_metrics": metrics,
        "u2_sigma_probability_metrics": u2_sigma_metrics,
        "u2_test_weighted_gaussian_nll": test_nll,
        "tail_probability_calibration": tail_calibration,
        "mean_layer_attention_by_horizon_channel": (
            layer_attention.mean(dim=0).tolist()
            if layer_attention is not None
            else None
        ),
        "s0_reference_metrics": base_summary["metrics"]["s0_conditional_mlp"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "metrics": metrics, "test_nll": test_nll, "fit": compact_fit}, indent=2))


if __name__ == "__main__":
    main()
