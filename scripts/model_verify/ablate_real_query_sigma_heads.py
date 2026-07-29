#!/usr/bin/env python3
"""Strict real-car ablation of Query residual Gaussian sigma heads.

The deterministic Query model is frozen and evaluated exactly once per split to
cache decoder hidden states.  Every sigma-head variant then sees the same
cached features, targets, file split, shuffle order, optimizer settings, and
training budget.  The untouched real test split is loaded only after all heads
have been selected on sigma validation data and calibrated on calibration data.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch import nn


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
from analyze_query_probability_shift import build_real_loader, load_query, read_paths
from train_kinematic_residual_ablation import (
    LOSS_WEIGHTS,
    prepare_batch,
    prepare_nominal_query,
)
from visualize_nuplan_query_probability_error import (
    CHANNEL_NAMES,
    channel_metrics,
    equal_count_bins,
    selective_risk,
)


PROTOCOL = "real_query_sigma_head_ablation_v1"
VARIANTS = (
    "linear_hidden",
    "mlp_hidden",
    "mlp_action_context",
    "mlp_action_context_nominal",
)
VARIANT_LABELS = {
    "linear_hidden": "Linear(hidden)",
    "mlp_hidden": "MLP(hidden)",
    "mlp_action_context": "MLP(hidden + action + context)",
    "mlp_action_context_nominal": "MLP(hidden + action + context + nominal)",
}
COLORS = {
    "linear_hidden": "#4C78A8",
    "mlp_hidden": "#F58518",
    "mlp_action_context": "#54A24B",
    "mlp_action_context_nominal": "#E45756",
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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bins", type=int, default=12)
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
        default=REPO_ROOT / "outputs/formal_real_query_sigma_head_ablation",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args):
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if len(variants) != len(set(variants)) or not variants:
        raise ValueError("--variants must contain unique supported names")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size < 1 or args.feature_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.hidden_dim < 1 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("invalid hidden dimension or dropout")
    if not 0.0 < args.lr_decay <= 1.0:
        raise ValueError("--lr-decay must be in (0, 1]")
    return variants


class ConditionalSigmaHead(nn.Module):
    """Linear or two-layer scale head over frozen decoder/covariate features."""

    def __init__(
        self,
        variant,
        latent_dim,
        output_dim,
        hidden_dim,
        dropout,
        sigma_floor,
    ):
        super().__init__()
        self.variant = variant
        self.sigma_floor = sigma_floor
        condition_dim = {
            "linear_hidden": 0,
            "mlp_hidden": 0,
            "mlp_action_context": 2 + 4,
            "mlp_action_context_nominal": 2 + 4 + 5 + 4,
        }[variant]
        input_dim = latent_dim + condition_dim
        if variant == "linear_hidden":
            self.network = nn.Linear(input_dim, output_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    @property
    def output_layer(self):
        return self.network if self.variant == "linear_hidden" else self.network[-1]

    def initialize_constant_sigma(self, sigma):
        adjusted = sigma.to(self.output_layer.bias) - self.sigma_floor
        if torch.any(adjusted <= 0):
            raise ValueError("initial sigma must exceed sigma floor")
        with torch.no_grad():
            self.output_layer.weight.zero_()
            self.output_layer.bias.copy_(inverse_softplus(adjusted))

    def forward(self, batch):
        parts = [batch["hidden"]]
        if self.variant in (
            "mlp_action_context",
            "mlp_action_context_nominal",
        ):
            context = batch["context"][:, None, :].expand(
                -1, batch["hidden"].shape[1], -1
            )
            parts.extend((batch["action"], context))
        if self.variant == "mlp_action_context_nominal":
            parts.extend((batch["nominal_state"], batch["nominal_transition"]))
        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return positive_scale(self.network(features), self.sigma_floor)


def split_files(manifest_dir):
    val_files = read_paths(manifest_dir / "val_files.txt")
    test_files = read_paths(manifest_dir / "test_files.txt")
    first = len(val_files) // 2
    second = first + len(val_files) // 4
    splits = {
        "sigma_train": val_files[:first],
        "sigma_val": val_files[first:second],
        "calibration": val_files[second:],
        "real_test": test_files,
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise RuntimeError(f"{left}/{right} overlap by {len(overlap)} files")
    return splits


def extract_cache(model, loader, stats, device, name):
    chunks = {
        key: []
        for key in (
            "hidden",
            "action",
            "context",
            "nominal_state",
            "nominal_transition",
            "error",
        )
    }
    model.eval()
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
            nominal_state, nominal_transition = prepare_nominal_query(
                batch, stats, device
            )
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
            hidden = model.transformer_decoder(
                tgt=action_emb,
                memory=history_emb,
                tgt_mask=model.tgt_mask,
                tgt_key_padding_mask=mask.to(device) if mask is not None else None,
            )
            mean = model.embedding["output"](hidden)
            target = batch["residual_target"].to(device)
            target = (target - stats["residual"][0]) / stats["residual"][1]
            values = {
                "hidden": hidden,
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
        f"hidden={tuple(cache['hidden'].shape)}",
        flush=True,
    )
    return cache


def cache_to(cache, device):
    return {key: value.to(device) for key, value in cache.items()}


def batch_from(cache, selection):
    return {key: value[selection] for key, value in cache.items()}


def evaluate_nll(head, cache, batch_size, weights, include_constant=False):
    head.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            batch = batch_from(cache, slice(start, start + batch_size))
            sigma = head(batch)
            loss = gaussian_nll(
                batch["error"],
                sigma,
                weights=weights,
                include_constant=include_constant,
            )
            examples = batch["error"].shape[0]
            total += loss.item() * examples
            count += examples
    return total / max(count, 1)


def collect_sigma(head, cache, batch_size):
    head.eval()
    values = []
    with torch.no_grad():
        for start in range(0, cache["error"].shape[0], batch_size):
            batch = batch_from(cache, slice(start, start + batch_size))
            values.append(head(batch).cpu())
    return torch.cat(values, dim=0)


def fit_head(variant, train, val, args, device):
    set_seed(args.seed)
    head = ConditionalSigmaHead(
        variant=variant,
        latent_dim=train["hidden"].shape[-1],
        output_dim=train["error"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        sigma_floor=args.sigma_floor,
    ).to(device)
    initial_sigma = torch.sqrt(
        torch.mean(train["error"].square(), dim=(0, 1))
    ).clamp_min(args.sigma_floor * 1.01)
    head.initialize_constant_sigma(initial_sigma)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay
    )
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
            sigma = head(batch)
            loss = gaussian_nll(batch["error"], sigma, weights=weights)
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
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
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
        "initial_sigma_normalized": initial_sigma.detach().cpu().tolist(),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "history": history,
        "state_dict": best_state,
    }


def calibrate_head(head, cache, args):
    sigma = collect_sigma(head, cache, args.batch_size)
    error = cache["error"].detach().cpu()
    temperature = horizon_channel_temperature(error, sigma)
    return temperature


def run_metrics(error, sigma, bins):
    error_np = error.numpy().astype(np.float64)
    sigma_np = sigma.numpy().astype(np.float64)
    channels = {
        channel: channel_metrics(
            error_np[..., index].reshape(-1),
            sigma_np[..., index].reshape(-1),
            bins,
        )
        for index, channel in enumerate(CHANNEL_NAMES)
    }
    aggregate_keys = (
        "spearman_sigma_abs_error",
        "top10_error_auc",
        "top10_error_recall_by_top10_sigma",
        "ence",
        "calibration_mae",
        "coverage_68",
        "coverage_90",
        "coverage_95",
    )
    aggregate = {
        key: float(np.nanmean([channels[channel][key] for channel in CHANNEL_NAMES]))
        for key in aggregate_keys
    }
    return {"aggregate": aggregate, "channels": channels}


def plot_training(fits, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant, fit in fits.items():
        history = fit["history"]
        train = [entry for entry in history if "train_nll" in entry]
        axes[0].plot(
            [entry["epoch"] for entry in train],
            [entry["train_nll"] for entry in train],
            color=COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
        axes[1].plot(
            [entry["epoch"] for entry in history],
            [entry["val_nll"] for entry in history],
            color=COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
    axes[0].set(title="Sigma-head train NLL", xlabel="Epoch", ylabel="Weighted NLL")
    axes[1].set(title="Sigma-head validation NLL", xlabel="Epoch", ylabel="Weighted NLL")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "01_training_nll.png", dpi=180)
    plt.close(fig)


def plot_aggregate(metrics, output):
    specifications = (
        ("spearman_sigma_abs_error", "Spearman", True),
        ("top10_error_auc", "Top-10% error AUC", True),
        ("calibration_mae", "Calibration MAE", False),
        ("ence", "ENCE", False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for axis, (key, title, higher) in zip(axes, specifications):
        variants = list(metrics)
        values = [metrics[variant]["aggregate"][key] for variant in variants]
        axis.bar(
            range(len(variants)),
            values,
            color=[COLORS[variant] for variant in variants],
        )
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=25, ha="right", fontsize=8)
        axis.set_title(f"{title}\n({'higher' if higher else 'lower'} is better)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Strict sigma-head ablation on untouched real test")
    fig.tight_layout()
    fig.savefig(output / "02_aggregate_metrics.png", dpi=180)
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
                label=VARIANT_LABELS[variant],
            )
        axis.set(title=channel, xlabel="Predicted-sigma quantile bin")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed physical RMSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Does higher predicted sigma separate higher actual error?")
    fig.tight_layout()
    fig.savefig(output / "03_sigma_rank_bins.png", dpi=180)
    plt.close(fig)


def plot_predicted_observed(arrays, bins, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        maximum = 0.0
        for variant, values in arrays.items():
            predicted, observed = equal_count_bins(
                values["sigma"][..., channel_index].reshape(-1),
                values["error"][..., channel_index].reshape(-1),
                bins,
            )
            maximum = max(maximum, float(predicted.max()), float(observed.max()))
            axis.plot(
                predicted,
                observed,
                marker="o",
                markersize=3,
                color=COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=1)
        axis.set(title=channel, xlabel="Predicted RMS sigma")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed RMSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Sigma-bin calibration")
    fig.tight_layout()
    fig.savefig(output / "04_predicted_vs_observed_bins.png", dpi=180)
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
                label=VARIANT_LABELS[variant],
            )
        axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
        axis.set(title=channel, xlabel="Fraction retained (lowest sigma first)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE / full-set RMSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Selective risk")
    fig.tight_layout()
    fig.savefig(output / "05_selective_risk.png", dpi=180)
    plt.close(fig)


def plot_reliability(arrays, output):
    nominal = np.linspace(0.05, 0.99, 20)
    thresholds = norm.ppf((1.0 + nominal) / 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharex=True, sharey=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        axis.plot(nominal, nominal, "k--", linewidth=1)
        for variant, values in arrays.items():
            z = (
                values["error"][..., channel_index].reshape(-1)
                / values["sigma"][..., channel_index].reshape(-1)
            )
            empirical = [np.mean(np.abs(z) <= threshold) for threshold in thresholds]
            axis.plot(
                nominal,
                empirical,
                color=COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        axis.set(title=channel, xlabel="Nominal coverage")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Horizon-channel calibrated reliability")
    fig.tight_layout()
    fig.savefig(output / "06_reliability.png", dpi=180)
    plt.close(fig)


def write_csv(metrics, output):
    fields = ["variant", "scope", "channel", "metric", "value"]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, result in metrics.items():
            for metric, value in result["aggregate"].items():
                writer.writerow(
                    {"variant": variant, "scope": "aggregate", "channel": "mean", "metric": metric, "value": value}
                )
            for channel, values in result["channels"].items():
                for metric, value in values.items():
                    writer.writerow(
                        {"variant": variant, "scope": "channel", "channel": channel, "metric": metric, "value": value}
                    )


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

    checkpoint, model_args, model, stats = load_query(
        args.mean_checkpoint.resolve(), device
    )
    params = KinematicBicycleParams(**checkpoint["params"])
    dataset_args = SimpleNamespace(**vars(args))
    dataset_args.eval_batch_size = args.feature_batch_size

    datasets = {}
    audits = {}
    caches_cpu = {}
    for name in ("sigma_train", "sigma_val", "calibration"):
        datasets[name], loader, audits[name] = build_real_loader(
            splits[name],
            model_args,
            params,
            dataset_args,
            name,
            shuffle=False,
        )
        caches_cpu[name] = extract_cache(model, loader, stats, device, name)

    caches = {name: cache_to(cache, device) for name, cache in caches_cpu.items()}
    initial_error = caches["sigma_train"]["error"]
    print(
        "initial normalized residual-error RMS:",
        torch.sqrt(torch.mean(initial_error.square(), dim=(0, 1))).tolist(),
        flush=True,
    )

    fitted = {}
    for variant in variants:
        head, fit = fit_head(
            variant, caches["sigma_train"], caches["sigma_val"], args, device
        )
        temperature = calibrate_head(head, caches["calibration"], args)
        fit["temperature"] = temperature.tolist()
        fitted[variant] = {"head": head, "fit": fit}
        torch.save(
            {
                "protocol": PROTOCOL,
                "variant": variant,
                "sigma_head_state_dict": fit["state_dict"],
                "temperature": temperature,
                "mean_checkpoint": str(args.mean_checkpoint.resolve()),
                "args": vars(args),
            },
            output / f"{variant}_best.pt",
        )

    # The held-out real test is first materialized only after every variant has
    # been selected and calibrated.  Nothing below feeds back into training.
    datasets["real_test"], test_loader, audits["real_test"] = build_real_loader(
        splits["real_test"],
        model_args,
        params,
        dataset_args,
        "real_test",
        shuffle=False,
    )
    test_cpu = extract_cache(model, test_loader, stats, device, "real_test")
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
        metrics[variant] = run_metrics(
            physical_error, physical_sigma, args.bins
        )
        metrics[variant]["weighted_gaussian_nll_normalized"] = float(
            gaussian_nll(
                test["error"],
                calibrated_sigma.to(device),
                weights=weights,
                include_constant=True,
            ).item()
        )
        metrics[variant]["unweighted_gaussian_nll_normalized"] = float(
            gaussian_nll(
                test["error"],
                calibrated_sigma.to(device),
                include_constant=True,
            ).item()
        )

    compact_fits = {}
    for variant, values in fitted.items():
        compact_fits[variant] = {
            key: value for key, value in values["fit"].items() if key != "state_dict"
        }
    plot_training(compact_fits, output)
    plot_aggregate(metrics, output)
    plot_sigma_bins(arrays, args.bins, output)
    plot_predicted_observed(arrays, args.bins, output)
    plot_selective(arrays, output)
    plot_reliability(arrays, output)
    write_csv(metrics, output)
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
            "same_cached_decoder_hidden": True,
            "same_split": True,
            "same_seed_and_shuffle_order": True,
            "same_optimizer_and_budget": True,
            "selection_uses_sigma_validation_only": True,
            "calibration_uses_calibration_only": True,
            "test_loaded_after_all_selection_and_calibration": True,
        },
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
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
