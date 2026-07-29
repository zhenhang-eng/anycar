#!/usr/bin/env python3
"""Evaluate Query residual uncertainty on simulation and real-car domains.

The three reported conditions mirror the legacy Transformer baseline:

* simulation mean/sigma model on simulation test data;
* simulation mean/sigma model transferred directly to real test data;
* real-fine-tuned mean/sigma model on the same real test data.

The real sigma head uses only the real validation manifest.  Its first half is
used for fitting, the next quarter for checkpoint selection, and the final
quarter for horizon/channel calibration.  The session-isolated real test
manifest is untouched until final evaluation.
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

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, str(REPO_ROOT / package_dir))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from car_foundation.kinematic_residual import KinematicBicycleParams
from car_foundation.probabilistic_residual import (
    FrozenMeanGaussianResidual,
    channel_temperature,
    horizon_channel_temperature,
)
from finetune_real_kinematic_residual import build_filtered_view
from train_kinematic_residual_ablation import make_model
from validate_nuplan_probabilistic_query import (
    CHANNELS,
    checkpoint_stats,
    collect_predictions,
    initialize_sigma,
    plot_history,
    run_epoch,
    set_seed,
)
from visualize_nuplan_query_probability_error import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline-style sim/real probability comparison for Query residual."
    )
    parser.add_argument("--simulation-mean-checkpoint", type=Path, required=True)
    parser.add_argument("--simulation-probability-run", type=Path, required=True)
    parser.add_argument("--simulation-large-holdout", type=Path, required=True)
    parser.add_argument("--real-mean-checkpoint", type=Path, required=True)
    parser.add_argument("--real-split-manifest-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "formal_query_probability_shift",
    )
    return parser.parse_args()


def read_paths(path):
    paths = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"No paths in {path}")
    return paths


def load_query(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("variant") != "query":
        raise ValueError(f"Expected Query checkpoint, got {checkpoint.get('variant')}")
    model_args = SimpleNamespace(**checkpoint["args"])
    model_args.device = str(device)
    model = make_model(model_args, device, "query")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model_args, model, checkpoint_stats(checkpoint, device)


def real_dataset_args(model_args, max_episodes):
    return SimpleNamespace(
        history_length=int(model_args.history_length),
        prediction_length=int(model_args.prediction_length),
        quaternion_norm_tolerance=0.01,
        position_jump_threshold=2.0,
        yaw_consistency_threshold=0.01,
        dt=0.05,
        steer_shift=1,
        max_episodes_per_split=max_episodes,
    )


def build_real_loader(files, model_args, params, args, name, shuffle=False):
    dataset, audit = build_filtered_view(
        files,
        real_dataset_args(model_args, args.max_episodes_per_split),
        params,
        name,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if shuffle else args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers if shuffle else 0,
        persistent_workers=shuffle and args.num_workers > 0,
    )
    return dataset, loader, audit


def physical_arrays(error, sigma, residual_std):
    scale = residual_std.detach().cpu().view(1, 1, -1)
    return {
        "error": (error * scale).numpy(),
        "sigma": (sigma * scale).numpy(),
    }


def fit_real_probability_head(
    checkpoint,
    model,
    stats,
    loaders,
    args,
    device,
    output_dir,
):
    wrapper = FrozenMeanGaussianResidual(
        model, output_dim=len(CHANNELS), sigma_floor=args.sigma_floor
    ).to(device)
    initial_sigma = initialize_sigma(
        wrapper, loaders["sigma_train"], stats, device
    )
    initial_val = run_epoch(
        wrapper, loaders["sigma_val"], stats, device, optimizer=None
    )
    optimizer = torch.optim.AdamW(
        wrapper.sigma_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay
    )
    best_path = output_dir / "real_probabilistic_query_best.pt"
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in wrapper.sigma_head.state_dict().items()
    }
    best_val = initial_val
    best_epoch = 0
    stale = 0
    history = [{"epoch": 0, "val_nll": initial_val, "learning_rate": args.lr}]
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_nll = run_epoch(
            wrapper, loaders["sigma_train"], stats, device, optimizer
        )
        val_nll = run_epoch(
            wrapper, loaders["sigma_val"], stats, device, optimizer=None
        )
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"real sigma epoch={epoch} train={train_nll:.7f} "
            f"val={val_nll:.7f} lr={learning_rate:.8f}",
            flush=True,
        )
        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in wrapper.sigma_head.state_dict().items()
            }
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                break
        scheduler.step()

    wrapper.sigma_head.load_state_dict(best_state)
    torch.save(
        {
            "sigma_head_state_dict": best_state,
            "initial_sigma": initial_sigma.cpu(),
            "epoch": best_epoch,
            "val_nll": best_val,
            "mean_checkpoint": str(args.real_mean_checkpoint.resolve()),
        },
        best_path,
    )
    calibration_error, calibration_sigma, _ = collect_predictions(
        wrapper, loaders["calibration"], stats, device
    )
    global_temperature = channel_temperature(
        calibration_error, calibration_sigma
    )
    horizon_temperature = horizon_channel_temperature(
        calibration_error, calibration_sigma
    )
    test_error, test_sigma, maximum_mean_difference = collect_predictions(
        wrapper,
        loaders["real_test"],
        stats,
        device,
        compare_mean=True,
    )
    calibrated_test_sigma = test_sigma * horizon_temperature.view(
        1, *horizon_temperature.shape
    )
    plot_history(history, output_dir / "real_sigma_training_curve.png")
    return (
        physical_arrays(test_error, calibrated_test_sigma, stats["residual"][1]),
        {
            "checkpoint": str(best_path),
            "best_epoch": best_epoch,
            "stopped_epoch": history[-1]["epoch"],
            "best_val_nll": best_val,
            "initial_sigma_normalized": initial_sigma.tolist(),
            "global_temperature": global_temperature.tolist(),
            "horizon_temperature": horizon_temperature.tolist(),
            "maximum_mean_difference": maximum_mean_difference,
            "history": history,
        },
        wrapper.sigma_head.state_dict(),
    )


def write_metrics(metrics, output_dir):
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fields = [
        "condition",
        "channel",
        *next(iter(next(iter(metrics.values())).values())).keys(),
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, channels in metrics.items():
            for channel, values in channels.items():
                writer.writerow({"condition": condition, "channel": channel, **values})


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    manifests = args.real_split_manifest_dir.resolve()
    val_files = read_paths(manifests / "val_files.txt")
    test_files = read_paths(manifests / "test_files.txt")
    first = len(val_files) // 2
    second = first + len(val_files) // 4
    real_splits = {
        "sigma_train": val_files[:first],
        "sigma_val": val_files[first:second],
        "calibration": val_files[second:],
        "real_test": test_files,
    }
    for index, left in enumerate(real_splits):
        for right in list(real_splits)[index + 1 :]:
            overlap = set(real_splits[left]) & set(real_splits[right])
            if overlap:
                raise RuntimeError(f"{left}/{right} overlap by {len(overlap)} files")
    for name, files in real_splits.items():
        (output_dir / f"{name}_files.txt").write_text("\n".join(files) + "\n")

    sim_checkpoint, sim_model_args, sim_model, sim_stats = load_query(
        args.simulation_mean_checkpoint, device
    )
    real_checkpoint, real_model_args, real_model, real_stats = load_query(
        args.real_mean_checkpoint, device
    )
    if sim_checkpoint["params"] != real_checkpoint["params"]:
        raise ValueError("Simulation and real checkpoints have different kinematic params")
    params = KinematicBicycleParams(**sim_checkpoint["params"])
    if (
        int(sim_model_args.history_length) != int(real_model_args.history_length)
        or int(sim_model_args.prediction_length) != int(real_model_args.prediction_length)
    ):
        raise ValueError("Simulation and real checkpoints use different sequence lengths")

    datasets = {}
    loaders = {}
    audits = {}
    for name, files in real_splits.items():
        datasets[name], loaders[name], audits[name] = build_real_loader(
            files,
            real_model_args,
            params,
            args,
            name,
            shuffle=name == "sigma_train",
        )

    large_summary = json.loads(
        (args.simulation_large_holdout / "summary.json").read_text()
    )
    sim_data = np.load(args.simulation_large_holdout / "confirmation_data.npz")
    sim_error = torch.from_numpy(sim_data["confirm_test_error"])
    sim_sigma = torch.from_numpy(sim_data["confirm_test_sigma"])
    sim_horizon_temperature = torch.tensor(
        large_summary["horizon_temperature"]["values"], dtype=sim_sigma.dtype
    )
    sim_on_sim = physical_arrays(
        sim_error,
        sim_sigma * sim_horizon_temperature.view(1, *sim_horizon_temperature.shape),
        sim_stats["residual"][1],
    )

    sim_wrapper = FrozenMeanGaussianResidual(
        sim_model, output_dim=len(CHANNELS), sigma_floor=args.sigma_floor
    ).to(device)
    sim_probability = torch.load(
        args.simulation_probability_run / "probabilistic_query_best.pt",
        map_location=device,
        weights_only=False,
    )
    sim_wrapper.sigma_head.load_state_dict(sim_probability["sigma_head_state_dict"])
    sim_real_error, sim_real_sigma, sim_real_mean_difference = collect_predictions(
        sim_wrapper,
        loaders["real_test"],
        sim_stats,
        device,
        compare_mean=True,
    )
    sim_on_real = physical_arrays(
        sim_real_error,
        sim_real_sigma
        * sim_horizon_temperature.to(device).view(
            1, *sim_horizon_temperature.shape
        ).cpu(),
        sim_stats["residual"][1],
    )

    adapted_on_real, real_fit, real_head_state = fit_real_probability_head(
        real_checkpoint,
        real_model,
        real_stats,
        loaders,
        args,
        device,
        output_dir,
    )
    raw = {
        "simulation_on_simulation": sim_on_sim,
        "simulation_on_real": sim_on_real,
        "adapted_real_on_real": adapted_on_real,
    }
    metrics = compute_metrics(raw, args.bins)
    write_metrics(metrics, output_dir)
    np.savez_compressed(
        output_dir / "probability_error_data.npz",
        **{
            f"{condition}_{kind}": values
            for condition, arrays in raw.items()
            for kind, values in arrays.items()
        },
        channel_names=np.asarray(CHANNELS),
    )
    torch.save(
        {
            "simulation_sigma_head": sim_wrapper.sigma_head.state_dict(),
            "simulation_horizon_temperature": sim_horizon_temperature,
            "real_sigma_head": real_head_state,
            "real_horizon_temperature": torch.tensor(
                real_fit["horizon_temperature"]
            ),
        },
        output_dir / "probability_heads.pt",
    )
    summary = {
        "protocol": "query_residual_sim_real_probability_shift_v1",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": {
            "simulation_on_simulation": "simulation Query mean/sigma on held-out simulation",
            "simulation_on_real": "same simulation Query mean/sigma transferred to session-isolated real test",
            "adapted_real_on_real": "real-fine-tuned Query mean and validation-fitted sigma on same real test",
        },
        "simulation_mean_checkpoint": str(args.simulation_mean_checkpoint.resolve()),
        "simulation_probability_checkpoint": str((args.simulation_probability_run / "probabilistic_query_best.pt").resolve()),
        "real_mean_checkpoint": str(args.real_mean_checkpoint.resolve()),
        "real_probability_fit": real_fit,
        "simulation_on_real_maximum_mean_difference": sim_real_mean_difference,
        "real_split_file_counts": {name: len(files) for name, files in real_splits.items()},
        "real_split_episode_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "real_split_audits": audits,
        "metrics": metrics,
        "data": str((output_dir / "probability_error_data.npz").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
