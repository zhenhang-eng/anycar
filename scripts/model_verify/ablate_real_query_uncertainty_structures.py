#!/usr/bin/env python3
"""Strict S0-S5 structural ablation for real Query residual uncertainty.

All variants consume the same frozen mean-model features and the same explicit
conditions: future action, current context, nominal state, and nominal
transition.  The only incremental changes are:

* S0: final decoder hidden + conditions through a two-layer MLP;
* S1: learned fusion of decoder layers 1/2/3 + conditional FiLM;
* S2: S1 + two causal depthwise-separable temporal convolution blocks;
* S3: S2 + independent output tower for each residual channel;
* S4: concatenated decoder layers + a deep causal Transformer encoder;
* S5: S4-style queries + a deep uncertainty decoder that cross-attends to the
  frozen mean model's compressed history memory.

The deterministic model is never updated.  Sigma validation selects each
variant, an independent calibration split fits 50x4 temperatures, and the real
test split is materialized only after all variants have been selected.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    run_metrics,
    set_seed,
    split_files,
)
from analyze_query_probability_shift import build_real_loader, load_query
from train_kinematic_residual_ablation import (
    LOSS_WEIGHTS,
    prepare_batch,
    prepare_nominal_query,
)
from visualize_nuplan_query_probability_error import (
    CHANNEL_NAMES,
    equal_count_bins,
    selective_risk,
)


PROTOCOL = "real_query_uncertainty_structure_ablation_s0_s5_v2"
VARIANTS = (
    "s0_conditional_mlp",
    "s1_film",
    "s2_film_tcn",
    "s3_film_tcn_towers",
    "s4_concat_transformer",
    "s5_history_cross_attention",
)
LABELS = {
    "s0_conditional_mlp": "S0: conditional MLP",
    "s1_film": "S1: layer fusion + FiLM",
    "s2_film_tcn": "S2: S1 + causal TCN",
    "s3_film_tcn_towers": "S3: S2 + channel towers",
    "s4_concat_transformer": "S4: concat + 4-layer Transformer",
    "s5_history_cross_attention": "S5: history cross-attention",
}
COLORS = {
    "s0_conditional_mlp": "#4C78A8",
    "s1_film": "#F58518",
    "s2_film_tcn": "#54A24B",
    "s3_film_tcn_towers": "#E45756",
    "s4_concat_transformer": "#B279A2",
    "s5_history_cross_attention": "#FF9DA6",
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
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tower-dim", type=int, default=64)
    parser.add_argument("--upper-dim", type=int, default=256)
    parser.add_argument("--upper-heads", type=int, default=8)
    parser.add_argument("--upper-layers", type=int, default=4)
    parser.add_argument("--upper-ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=0,
        help="0 uses every valid episode; positive values are for smoke tests.",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/formal_real_query_uncertainty_s0_s5",
    )
    return parser.parse_args()


def validate_args(args):
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or not variants or len(variants) != len(set(variants)):
        raise ValueError(f"Invalid variants: {variants}; unknown={unknown}")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size < 1 or args.feature_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.hidden_dim < 1 or args.tower_dim < 1 or args.upper_dim < 1:
        raise ValueError("hidden dimensions must be positive")
    if args.upper_heads < 1 or args.upper_layers < 1 or args.upper_ff_dim < 1:
        raise ValueError("upper Transformer dimensions must be positive")
    if args.upper_dim % args.upper_heads:
        raise ValueError("--upper-dim must be divisible by --upper-heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.lr_decay <= 1.0:
        raise ValueError("--lr-decay must be in (0, 1]")
    return variants


def explicit_condition(batch):
    context = batch["context"][:, None, :].expand(
        -1, batch["action"].shape[1], -1
    )
    return torch.cat(
        (
            batch["action"],
            context,
            batch["nominal_state"],
            batch["nominal_transition"],
        ),
        dim=-1,
    )


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        residual = value
        value = self.norm(value).transpose(1, 2)
        value = F.pad(value, (self.left_padding, 0))
        value = self.depthwise(value)
        value = self.pointwise(value).transpose(1, 2)
        return residual + self.dropout(F.silu(value))


class StructuralSigmaHead(nn.Module):
    def __init__(
        self,
        variant,
        latent_dim,
        condition_dim,
        output_dim,
        hidden_dim,
        tower_dim,
        dropout,
        sigma_floor,
        decoder_layers,
        sequence_length,
        upper_dim,
        upper_heads,
        upper_layers,
        upper_ff_dim,
    ):
        super().__init__()
        self.variant = variant
        self.sigma_floor = sigma_floor
        self.output_dim = output_dim
        self.layer_logits = None
        self.condition_film = None
        self.temporal = None
        self.shared_head = None
        self.channel_towers = None
        self.upper_input = None
        self.upper_position = None
        self.upper_encoder = None
        self.upper_decoder = None
        self.history_projection = None

        if variant == "s0_conditional_mlp":
            self.shared_head = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        elif variant in (
            "s1_film",
            "s2_film_tcn",
            "s3_film_tcn_towers",
        ):
            self.layer_logits = nn.Parameter(torch.zeros(decoder_layers))
            self.fused_norm = nn.LayerNorm(latent_dim)
            self.condition_film = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )
            with torch.no_grad():
                self.condition_film[-1].weight.zero_()
                self.condition_film[-1].bias.zero_()
            if variant in ("s2_film_tcn", "s3_film_tcn_towers"):
                self.temporal = nn.Sequential(
                    CausalTCNBlock(latent_dim, kernel_size=5, dilation=1, dropout=dropout),
                    CausalTCNBlock(latent_dim, kernel_size=5, dilation=2, dropout=dropout),
                )
            if variant == "s3_film_tcn_towers":
                self.channel_towers = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(latent_dim, tower_dim),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(tower_dim, 1),
                        )
                        for _ in range(output_dim)
                    ]
                )
            else:
                self.shared_head = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
        else:
            # S4/S5 intentionally retain the full per-layer representation
            # instead of reducing it to the scalar-weighted sum used by S1-S3.
            self.upper_input = nn.Sequential(
                nn.Linear(latent_dim * decoder_layers + condition_dim, upper_dim),
                nn.LayerNorm(upper_dim),
                nn.SiLU(),
            )
            self.upper_position = nn.Parameter(torch.zeros(sequence_length, upper_dim))
            nn.init.normal_(self.upper_position, std=0.02)
            if variant == "s4_concat_transformer":
                layer = nn.TransformerEncoderLayer(
                    d_model=upper_dim,
                    nhead=upper_heads,
                    dim_feedforward=upper_ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.upper_encoder = nn.TransformerEncoder(
                    layer,
                    num_layers=upper_layers,
                    norm=nn.LayerNorm(upper_dim),
                    enable_nested_tensor=False,
                )
            elif variant == "s5_history_cross_attention":
                self.history_projection = nn.Linear(latent_dim, upper_dim)
                layer = nn.TransformerDecoderLayer(
                    d_model=upper_dim,
                    nhead=upper_heads,
                    dim_feedforward=upper_ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.upper_decoder = nn.TransformerDecoder(
                    layer,
                    num_layers=upper_layers,
                    norm=nn.LayerNorm(upper_dim),
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")
            self.register_buffer(
                "upper_causal_mask",
                nn.Transformer.generate_square_subsequent_mask(sequence_length),
                persistent=False,
            )
            self.channel_towers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(upper_dim, tower_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        nn.Linear(tower_dim, 1),
                    )
                    for _ in range(output_dim)
                ]
            )

    def initialize_constant_sigma(self, sigma):
        adjusted = sigma - self.sigma_floor
        if torch.any(adjusted <= 0):
            raise ValueError("initial sigma must exceed sigma floor")
        bias = inverse_softplus(adjusted)
        with torch.no_grad():
            if self.channel_towers is not None:
                for index, tower in enumerate(self.channel_towers):
                    tower[-1].weight.zero_()
                    tower[-1].bias.copy_(bias[index : index + 1])
            else:
                self.shared_head[-1].weight.zero_()
                self.shared_head[-1].bias.copy_(bias)

    def uncertainty_feature(self, batch):
        layers = batch["hidden_layers"]
        weights = torch.softmax(self.layer_logits, dim=0)
        value = torch.sum(
            layers * weights.view(1, 1, -1, 1), dim=2
        )
        value = self.fused_norm(value)
        film = self.condition_film(explicit_condition(batch))
        gamma, beta = torch.chunk(film, 2, dim=-1)
        value = value * (1.0 + gamma) + beta
        if self.temporal is not None:
            value = self.temporal(value)
        return value

    def forward(self, batch):
        condition = explicit_condition(batch)
        if self.variant == "s0_conditional_mlp":
            value = torch.cat((batch["hidden_layers"][:, :, -1], condition), dim=-1)
            raw = self.shared_head(value)
        elif self.variant in (
            "s1_film",
            "s2_film_tcn",
            "s3_film_tcn_towers",
        ):
            value = self.uncertainty_feature(batch)
            if self.channel_towers is not None:
                raw = torch.cat([tower(value) for tower in self.channel_towers], dim=-1)
            else:
                raw = self.shared_head(value)
        else:
            layers = batch["hidden_layers"].flatten(start_dim=2)
            value = self.upper_input(torch.cat((layers, condition), dim=-1))
            value = value + self.upper_position[: value.shape[1]]
            mask = self.upper_causal_mask[: value.shape[1], : value.shape[1]]
            if self.upper_encoder is not None:
                value = self.upper_encoder(value, mask=mask, is_causal=True)
            else:
                memory = self.history_projection(batch["history_memory"])
                value = self.upper_decoder(
                    value,
                    memory,
                    tgt_mask=mask,
                    tgt_is_causal=True,
                )
            raw = torch.cat([tower(value) for tower in self.channel_towers], dim=-1)
        return positive_scale(raw, self.sigma_floor)


def decoder_layer_outputs(model, tgt, memory, tgt_key_padding_mask):
    output = tgt
    layers = []
    for layer in model.transformer_decoder.layers:
        output = layer(
            output,
            memory,
            tgt_mask=model.tgt_mask,
            tgt_key_padding_mask=(
                tgt_key_padding_mask.to(model.device)
                if tgt_key_padding_mask is not None
                else None
            ),
            tgt_is_causal=True,
        )
        layers.append(output)
    if model.transformer_decoder.norm is not None:
        output = model.transformer_decoder.norm(output)
        layers[-1] = output
    return torch.stack(layers, dim=2)


def extract_cache(model, loader, stats, device, name, audit_tolerance=1e-6):
    chunks = {
        key: []
        for key in (
            "hidden_layers",
            "history_memory",
            "action",
            "context",
            "nominal_state",
            "nominal_transition",
            "error",
        )
    }
    model.eval()
    maximum_hidden_difference = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            history, action, context, mask = prepare_batch(
                batch,
                stats["history"][0],
                stats["history"][1],
                stats["context"][0],
                stats["context"][1],
                device,
            )
            nominal_state, nominal_transition = prepare_nominal_query(batch, stats, device)
            history_emb = model.position_encoding["history"](
                model._build_history_emb(history)
            )
            action_emb = model.position_encoding["action"](
                model._build_action_emb(
                    history,
                    action,
                    context,
                    nominal_state,
                    nominal_transition,
                )
            )
            hidden_layers = decoder_layer_outputs(model, action_emb, history_emb, mask)
            if batch_index == 1:
                reference = model.transformer_decoder(
                    tgt=action_emb,
                    memory=history_emb,
                    tgt_mask=model.tgt_mask,
                    tgt_key_padding_mask=mask.to(device) if mask is not None else None,
                )
                maximum_hidden_difference = float(
                    torch.max(torch.abs(reference - hidden_layers[:, :, -1])).item()
                )
                if maximum_hidden_difference > audit_tolerance:
                    raise RuntimeError(
                        "manual decoder layers changed final hidden: "
                        f"max_abs={maximum_hidden_difference}"
                    )
            mean = model.embedding["output"](hidden_layers[:, :, -1])
            target = batch["residual_target"].to(device)
            target = (target - stats["residual"][0]) / stats["residual"][1]
            values = {
                "hidden_layers": hidden_layers,
                "history_memory": history_emb,
                "action": action,
                "context": context,
                "nominal_state": nominal_state,
                "nominal_transition": nominal_transition,
                "error": target - mean,
            }
            for key, value in values.items():
                chunks[key].append(value.detach().cpu())
            if batch_index % 10 == 0:
                print(f"cache {name}: {batch_index}/{len(loader)} batches", flush=True)
    cache = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    print(
        f"cache {name}: {cache['error'].shape[0]} episodes, "
        f"layers={tuple(cache['hidden_layers'].shape)}, "
        f"hidden_audit={maximum_hidden_difference:.3e}",
        flush=True,
    )
    return cache, maximum_hidden_difference


def evaluate_nll(head, cache, batch_size, weights):
    head.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            batch = batch_from(cache, slice(start, start + batch_size))
            loss = gaussian_nll(batch["error"], head(batch), weights=weights)
            examples = batch["error"].shape[0]
            total += loss.item() * examples
            count += examples
    return total / max(count, 1)


def collect_sigma(head, cache, batch_size):
    head.eval()
    values = []
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            values.append(head(batch_from(cache, slice(start, start + batch_size))).cpu())
    return torch.cat(values, dim=0)


def make_head(variant, cache, args, device):
    return StructuralSigmaHead(
        variant=variant,
        latent_dim=cache["hidden_layers"].shape[-1],
        condition_dim=15,
        output_dim=cache["error"].shape[-1],
        hidden_dim=args.hidden_dim,
        tower_dim=args.tower_dim,
        dropout=args.dropout,
        sigma_floor=args.sigma_floor,
        decoder_layers=cache["hidden_layers"].shape[2],
        sequence_length=cache["hidden_layers"].shape[1],
        upper_dim=args.upper_dim,
        upper_heads=args.upper_heads,
        upper_layers=args.upper_layers,
        upper_ff_dim=args.upper_ff_dim,
    ).to(device)


def fit_head(variant, train, val, args, device):
    set_seed(args.seed)
    head = make_head(variant, train, args, device)
    initial_sigma = torch.sqrt(
        torch.mean(train["error"].square(), dim=(0, 1))
    ).clamp_min(args.sigma_floor * 1.01)
    head.initialize_constant_sigma(initial_sigma)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    weights = LOSS_WEIGHTS.to(device)
    initial_val = evaluate_nll(head, val, args.batch_size, weights)
    best_val = initial_val
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
    }
    history = [{"epoch": 0, "val_nll": initial_val, "learning_rate": args.lr}]
    stale = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        head.train()
        learning_rate = optimizer.param_groups[0]["lr"]
        permutation = torch.randperm(train["error"].shape[0], generator=generator)
        total = 0.0
        count = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            batch = batch_from(train, indices)
            loss = gaussian_nll(batch["error"], head(batch), weights=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            examples = batch["error"].shape[0]
            total += loss.detach().item() * examples
            count += examples
        train_nll = total / max(count, 1)
        val_nll = evaluate_nll(head, val, args.batch_size, weights)
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"{variant} epoch={epoch:03d} train={train_nll:.7f} "
            f"val={val_nll:.7f} lr={learning_rate:.8f}",
            flush=True,
        )
        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break
        scheduler.step()
    head.load_state_dict(best_state)
    return head, {
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_val_nll": best_val,
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "initial_sigma_normalized": initial_sigma.detach().cpu().tolist(),
        "history": history,
        "state_dict": best_state,
    }


def benchmark_head(head, cache, batch_sizes, warmup, iterations, device):
    result = {}
    head.eval()
    with torch.no_grad():
        for requested in batch_sizes:
            batch_size = min(requested, cache["error"].shape[0])
            batch = batch_from(cache, slice(0, batch_size))
            for _ in range(warmup):
                head(batch)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                head(batch)
            end.record()
            torch.cuda.synchronize(device)
            milliseconds = start.elapsed_time(end) / iterations
            result[str(requested)] = {
                "actual_batch_size": batch_size,
                "milliseconds_per_batch": milliseconds,
                "microseconds_per_episode": milliseconds * 1000.0 / batch_size,
            }
    return result


def plot_training(fits, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant, fit in fits.items():
        history = fit["history"]
        train = [entry for entry in history if "train_nll" in entry]
        axes[0].plot(
            [entry["epoch"] for entry in train],
            [entry["train_nll"] for entry in train],
            color=COLORS[variant],
            label=LABELS[variant],
        )
        axes[1].plot(
            [entry["epoch"] for entry in history],
            [entry["val_nll"] for entry in history],
            color=COLORS[variant],
            label=LABELS[variant],
        )
    axes[0].set(title="Train weighted Gaussian NLL", xlabel="Epoch", ylabel="NLL")
    axes[1].set(title="Validation weighted Gaussian NLL", xlabel="Epoch", ylabel="NLL")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "01_training_nll.png", dpi=180)
    plt.close(fig)


def plot_channel_metrics(metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(CHANNEL_NAMES))
    width = 0.19
    for index, variant in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            [metrics[variant]["channels"][channel]["spearman_sigma_abs_error"] for channel in CHANNEL_NAMES],
            width,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        axes[1].bar(
            x + offset,
            [metrics[variant]["channels"][channel]["top10_error_auc"] for channel in CHANNEL_NAMES],
            width,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    for axis, title in zip(axes, ("Sigma-error Spearman", "Top-10% error AUC")):
        axis.set_xticks(x)
        axis.set_xticklabels(CHANNEL_NAMES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Random AUC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "02_channel_ranking_metrics.png", dpi=180)
    plt.close(fig)


def plot_aggregate(metrics, output):
    specs = (
        ("spearman_sigma_abs_error", "Mean Spearman", True),
        ("top10_error_auc", "Mean top-10% AUC", True),
        ("calibration_mae", "Calibration MAE", False),
        ("ence", "ENCE", False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    variants = list(metrics)
    for axis, (key, title, higher) in zip(axes, specs):
        axis.bar(
            range(len(variants)),
            [metrics[variant]["aggregate"][key] for variant in variants],
            color=[COLORS[variant] for variant in variants],
        )
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([variant.split("_")[0].upper() for variant in variants])
        axis.set_title(f"{title}\n({'higher' if higher else 'lower'} is better)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "03_aggregate_metrics.png", dpi=180)
    plt.close(fig)


def plot_sigma_bins(arrays, bins, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        for variant, values in arrays.items():
            error = values["error"][..., channel_index].reshape(-1)
            sigma = values["sigma"][..., channel_index].reshape(-1)
            groups = np.array_split(np.argsort(sigma), bins)
            observed = [np.sqrt(np.mean(np.square(error[group]))) for group in groups]
            axis.plot(
                np.arange(1, bins + 1),
                observed,
                marker="o",
                markersize=3,
                color=COLORS[variant],
                label=LABELS[variant],
            )
        axis.set(title=channel, xlabel="Predicted-sigma quantile bin")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed physical RMSE")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "04_sigma_rank_bins.png", dpi=180)
    plt.close(fig)


def plot_selective(arrays, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        for variant, values in arrays.items():
            fractions, risk = selective_risk(
                values["sigma"][..., channel_index].reshape(-1),
                values["error"][..., channel_index].reshape(-1),
            )
            axis.plot(
                fractions,
                risk / risk[-1],
                color=COLORS[variant],
                label=LABELS[variant],
            )
        axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
        axis.set(title=channel, xlabel="Fraction retained (lowest sigma first)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE / full-set RMSE")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "05_selective_risk.png", dpi=180)
    plt.close(fig)


def plot_latency(fits, output):
    variants = list(fits)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(
        range(len(variants)),
        [fits[v]["parameter_count"] / 1000.0 for v in variants],
        color=[COLORS[v] for v in variants],
    )
    axes[1].bar(
        range(len(variants)),
        [fits[v]["benchmark"]["256"]["milliseconds_per_batch"] for v in variants],
        color=[COLORS[v] for v in variants],
    )
    for axis, title, ylabel in (
        (axes[0], "Trainable uncertainty parameters", "Thousand parameters"),
        (axes[1], "Head-only GPU latency, batch=256", "Milliseconds / batch"),
    ):
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([v.split("_")[0].upper() for v in variants])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "06_parameters_latency.png", dpi=180)
    plt.close(fig)


def write_metrics_csv(metrics, output):
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "scope", "channel", "metric", "value"))
        for variant, result in metrics.items():
            for metric, value in result["aggregate"].items():
                writer.writerow((variant, "aggregate", "mean", metric, value))
            for channel, values in result["channels"].items():
                for metric, value in values.items():
                    writer.writerow((variant, "channel", channel, metric, value))


def main():
    args = parse_args()
    variants = validate_args(args)
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
        caches_cpu[name], hidden_audits[name] = extract_cache(
            model, loader, stats, device, name
        )
    caches = {name: cache_to(cache, device) for name, cache in caches_cpu.items()}

    fitted = {}
    for variant in variants:
        head, fit = fit_head(
            variant, caches["sigma_train"], caches["sigma_val"], args, device
        )
        calibration_sigma = collect_sigma(head, caches["calibration"], args.batch_size)
        calibration_error = caches_cpu["calibration"]["error"]
        temperature = horizon_channel_temperature(calibration_error, calibration_sigma)
        fit["temperature"] = temperature.tolist()
        fitted[variant] = {"head": head, "fit": fit}
        torch.save(
            {
                "protocol": PROTOCOL,
                "variant": variant,
                "state_dict": fit["state_dict"],
                "temperature": temperature,
                "mean_checkpoint": str(args.mean_checkpoint.resolve()),
                "args": vars(args),
            },
            output / f"{variant}_best.pt",
        )

    # The test split is not materialized until every S0-S3 variant is selected
    # and independently calibrated.
    datasets["real_test"], test_loader, audits["real_test"] = build_real_loader(
        splits["real_test"], model_args, params, dataset_args, "real_test", shuffle=False
    )
    test_cpu, hidden_audits["real_test"] = extract_cache(
        model, test_loader, stats, device, "real_test"
    )
    test = cache_to(test_cpu, device)
    physical_scale = stats["residual"][1].detach().cpu().view(1, 1, -1)
    physical_error = test_cpu["error"] * physical_scale
    arrays = {}
    metrics = {}
    weights = LOSS_WEIGHTS.to(device)
    for variant in variants:
        head = fitted[variant]["head"]
        raw_sigma = collect_sigma(head, test, args.batch_size)
        temperature = torch.tensor(fitted[variant]["fit"]["temperature"])
        calibrated_sigma = raw_sigma * temperature.view(1, *temperature.shape)
        physical_sigma = calibrated_sigma * physical_scale
        arrays[variant] = {
            "error": physical_error.numpy().astype(np.float64),
            "sigma": physical_sigma.numpy().astype(np.float64),
        }
        metrics[variant] = run_metrics(physical_error, physical_sigma, args.bins)
        metrics[variant]["weighted_gaussian_nll_normalized"] = float(
            gaussian_nll(
                test["error"],
                calibrated_sigma.to(device),
                weights=weights,
                include_constant=True,
            ).item()
        )
        fitted[variant]["fit"]["benchmark"] = benchmark_head(
            head,
            test,
            batch_sizes=(1, 256),
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
            device=device,
        )

    compact_fits = {
        variant: {
            key: value
            for key, value in values["fit"].items()
            if key != "state_dict"
        }
        for variant, values in fitted.items()
    }
    plot_training(compact_fits, output)
    plot_channel_metrics(metrics, output)
    plot_aggregate(metrics, output)
    plot_sigma_bins(arrays, args.bins, output)
    plot_selective(arrays, output)
    plot_latency(compact_fits, output)
    write_metrics_csv(metrics, output)
    np.savez_compressed(
        output / "test_probability_error_data.npz",
        error=physical_error.numpy(),
        **{f"sigma_{variant}": values["sigma"] for variant, values in arrays.items()},
        channel_names=np.asarray(CHANNEL_NAMES),
    )

    summary = {
        "protocol": PROTOCOL,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "mean_checkpoint": str(args.mean_checkpoint.resolve()),
        "mean_checkpoint_epoch": int(checkpoint["epoch"]),
        "strict_controls": {
            "same_explicit_inputs": [
                "future_action",
                "current_context",
                "nominal_state",
                "nominal_transition",
            ],
            "same_cached_decoder_layer_outputs": True,
            "same_split_seed_optimizer_and_budget": True,
            "mean_model_frozen": True,
            "test_loaded_after_all_selection_and_calibration": True,
        },
        "final_hidden_equivalence_max_abs": hidden_audits,
        "split_file_counts": {name: len(files) for name, files in splits.items()},
        "split_episode_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "split_audits": audits,
        "fits": compact_fits,
        "metrics": metrics,
        "artifacts": {
            "data": str((output / "test_probability_error_data.npz").resolve()),
            "metrics_csv": str((output / "metrics.csv").resolve()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "metrics": metrics, "fits": compact_fits}, indent=2))


if __name__ == "__main__":
    main()
