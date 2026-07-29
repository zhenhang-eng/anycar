#!/usr/bin/env python3
"""Fit frozen-mean Gaussian heads and quantify simulation-to-real shift.

This protocol is intentionally tied to checkpoints produced by
``train_transformer_pytorch.py``.  It fits uncertainty only on each run's
validation files, calibrates horizon/channel temperatures on a disjoint slice,
and keeps the run's test files untouched until final evaluation.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "car_foundation"))

from car_foundation.dataset import MujocoDataset
from car_foundation.models import TorchTransformerDecoder


CHANNEL_INDICES = (0, 1, 2, 3, 5)
CHANNEL_NAMES = ("dx_body", "dy_body", "dyaw", "dvx", "dyawrate")
COVERAGE = {"68": 1.0, "90": 1.6448536269514722, "95": 1.959963984540054}
SELECTED_HORIZONS = (1, 5, 10, 20, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-run", required=True)
    parser.add_argument("--real-run", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "transformer_probability_shift"),
    )
    return parser.parse_args()


def read_paths(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())
    checkpoint_path = Path(summary["best_val_checkpoint"] or summary["latest_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TorchTransformerDecoder(
        state_dim=6,
        action_dim=2,
        output_dim=6,
        latent_dim=256,
        num_heads=4,
        num_layers=3,
        device=device,
        dropout=0.1,
        history_length=250,
        prediction_length=50,
        compressed_history_length=42,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "dir": run_dir,
        "summary": summary,
        "checkpoint": checkpoint_path,
        "model": model,
        "mean": checkpoint["input_mean"].to(device),
        "std": checkpoint["input_std"].to(device),
        "val_files": read_paths(run_dir / "val_files.txt"),
        "test_files": read_paths(run_dir / "test_files.txt"),
    }


def make_loader(files, batch_size, shuffle):
    dataset = MujocoDataset(
        files,
        history_length=251,
        action_length=50,
        teacher_forcing=False,
        binary_mask=False,
        use_zero_point=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def hidden_mean_target(system, batch, device):
    history, action, target, mask = batch
    history = history.to(device)
    action = action.to(device)
    target = target.to(device)
    history = history.clone()
    history[..., :6] = (history[..., :6] - system["mean"]) / system["std"]
    history = history[:, 1:, :]
    target = (target[..., :6] - system["mean"]) / system["std"]
    model = system["model"]
    with torch.no_grad():
        memory = model.position_encoding["history"](
            model._build_history_emb(history)
        )
        query = model.position_encoding["action"](
            model.embedding["action"](action)
        )
        hidden = model.transformer_decoder(
            tgt=query,
            memory=memory,
            tgt_mask=model.tgt_mask,
            tgt_key_padding_mask=mask.to(device) if mask is not None else None,
        )
        prediction = model.embedding["output"](hidden)
    index = torch.tensor(CHANNEL_INDICES, device=device)
    return hidden.detach(), prediction.index_select(-1, index), target.index_select(-1, index)


def positive_sigma(head, hidden, floor):
    return floor + F.softplus(head(hidden))


def inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


def gaussian_nll(error, sigma):
    return 0.5 * ((error / sigma).square() + 2.0 * torch.log(sigma))


def initialize_head(system, head, loader, device, floor):
    squared = torch.zeros(len(CHANNEL_INDICES), device=device)
    count = 0
    for batch in loader:
        _, mean, target = hidden_mean_target(system, batch, device)
        squared += (target - mean).square().sum(dim=(0, 1))
        count += target.shape[0] * target.shape[1]
    initial = torch.sqrt(squared / count).clamp_min(floor * 2.0)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(inverse_softplus(initial - floor))
    return initial


def run_head_epoch(system, head, loader, device, floor, optimizer=None):
    total = 0.0
    count = 0
    for batch in loader:
        hidden, mean, target = hidden_mean_target(system, batch, device)
        sigma = positive_sigma(head, hidden, floor)
        loss = gaussian_nll(target - mean, sigma).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_count = target.numel()
        total += loss.detach().item() * batch_count
        count += batch_count
    return total / count


def fit_head(system, args, device):
    files = system["val_files"]
    first = len(files) // 2
    second = first + len(files) // 4
    splits = {
        "sigma_train": files[:first],
        "sigma_val": files[first:second],
        "calibration": files[second:],
    }
    loaders = {
        name: make_loader(paths, args.batch_size, name == "sigma_train")
        for name, paths in splits.items()
    }
    head = nn.Linear(256, len(CHANNEL_INDICES)).to(device)
    initial = initialize_head(
        system, head, loaders["sigma_train"], device, args.sigma_floor
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    best_val = run_head_epoch(
        system, head, loaders["sigma_val"], device, args.sigma_floor
    )
    history = [{"epoch": 0, "val_nll": best_val}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_nll = run_head_epoch(
            system, head, loaders["sigma_train"], device, args.sigma_floor, optimizer
        )
        val_nll = run_head_epoch(
            system, head, loaders["sigma_val"], device, args.sigma_floor
        )
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        print(f"{system['dir'].name}: sigma epoch={epoch} train={train_nll:.6f} val={val_nll:.6f}", flush=True)
        if val_nll < best_val:
            best_val = val_nll
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    error, sigma, _ = collect(system, head, loaders["calibration"], device, args.sigma_floor)
    temperature = torch.sqrt(torch.mean((error / sigma).square(), dim=0))
    return head, temperature, {
        "files": {name: len(value) for name, value in splits.items()},
        "initial_sigma": initial.tolist(),
        "best_val_nll": best_val,
        "history": history,
    }


def collect(system, head, loader, device, floor):
    errors, sigmas, features = [], [], []
    for batch in loader:
        history = batch[0]
        hidden, mean, target = hidden_mean_target(system, batch, device)
        sigma = positive_sigma(head, hidden, floor)
        errors.append((target - mean).cpu())
        sigmas.append(sigma.detach().cpu())
        action = batch[1]
        features.append(
            torch.stack(
                (
                    history[:, -1, 3],
                    history[:, -1, 5],
                    history[:, -1, 6],
                    history[:, -1, 7],
                    action[..., 0].mean(dim=1),
                    action[..., 0].std(dim=1),
                    action[..., 1].abs().mean(dim=1),
                    action[..., 1].abs().amax(dim=1),
                ),
                dim=-1,
            )
        )
    return torch.cat(errors), torch.cat(sigmas), torch.cat(features)


def metrics(error, sigma, physical_std):
    z = error / sigma
    physical_error = error * physical_std.view(1, 1, -1)
    nll = gaussian_nll(error, sigma)
    result = {
        "episodes": error.shape[0],
        "normalized_nll": float(nll.mean()),
        "physical_rmse": {},
        "z_mean": {},
        "z_rms": {},
        "z_excess_kurtosis": {},
        "coverage": {},
        "selected_horizons": {},
    }
    for index, channel in enumerate(CHANNEL_NAMES):
        values = z[..., index].numpy().reshape(-1)
        result["physical_rmse"][channel] = float(
            torch.sqrt(torch.mean(physical_error[..., index].square()))
        )
        result["z_mean"][channel] = float(np.mean(values))
        result["z_rms"][channel] = float(np.sqrt(np.mean(np.square(values))))
        result["z_excess_kurtosis"][channel] = float(kurtosis(values, fisher=True, bias=False))
    for label, threshold in COVERAGE.items():
        empirical = torch.mean((torch.abs(z) <= threshold).float(), dim=(0, 1))
        result["coverage"][label] = {
            channel: float(empirical[index]) for index, channel in enumerate(CHANNEL_NAMES)
        }
    for horizon in SELECTED_HORIZONS:
        hz = z[:, horizon - 1]
        result["selected_horizons"][str(horizon)] = {
            "z_mean": {
                channel: float(hz[:, index].mean())
                for index, channel in enumerate(CHANNEL_NAMES)
            },
            "z_rms": {
                channel: float(torch.sqrt(torch.mean(hz[:, index].square())))
                for index, channel in enumerate(CHANNEL_NAMES)
            },
        }
    return result


def evaluate(system, head, temperature, files, args, device):
    loader = make_loader(files, args.batch_size, False)
    error, sigma, features = collect(system, head, loader, device, args.sigma_floor)
    calibrated = sigma * temperature.view(1, 50, len(CHANNEL_INDICES))
    index = torch.tensor(CHANNEL_INDICES, device=system["std"].device)
    physical_std = system["std"].index_select(0, index).cpu()
    return metrics(error, calibrated, physical_std), features.numpy()


def standardized_mean_difference(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    variance = 0.5 * (np.var(left) + np.var(right))
    return float((np.mean(right) - np.mean(left)) / math.sqrt(variance)) if variance > 0 else 0.0


def main():
    args = parse_args()
    torch.manual_seed(3407)
    np.random.seed(3407)
    device = torch.device("cuda")
    simulation = load_run(args.simulation_run, device)
    real = load_run(args.real_run, device)

    sim_head, sim_temperature, sim_fit = fit_head(simulation, args, device)
    sim_test, sim_features = evaluate(
        simulation, sim_head, sim_temperature, simulation["test_files"], args, device
    )
    sim_to_real, real_features_under_sim = evaluate(
        simulation, sim_head, sim_temperature, real["test_files"], args, device
    )

    real_head, real_temperature, real_fit = fit_head(real, args, device)
    adapted_real, _ = evaluate(
        real, real_head, real_temperature, real["test_files"], args, device
    )

    feature_names = (
        "last_dvx", "last_dyawrate", "last_throttle", "last_steer",
        "future_mean_throttle", "future_throttle_std",
        "future_mean_abs_steer", "future_max_abs_steer",
    )
    feature_shift = {
        name: standardized_mean_difference(sim_features[:, index], real_features_under_sim[:, index])
        for index, name in enumerate(feature_names)
    }
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "transformer_sim_real_probability_shift_v1",
        "args": vars(args),
        "simulation_checkpoint": str(simulation["checkpoint"]),
        "real_checkpoint": str(real["checkpoint"]),
        "supervised_channels": list(CHANNEL_NAMES),
        "excluded_channel": "dvy (deterministic training weight is zero)",
        "simulation_probability_fit": sim_fit,
        "real_probability_fit": real_fit,
        "simulation_test": sim_test,
        "simulation_model_on_real_test": sim_to_real,
        "adapted_real_model_on_real_test": adapted_real,
        "simulation_to_real_feature_smd": feature_shift,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "simulation_sigma_head": sim_head.state_dict(),
            "simulation_temperature": sim_temperature,
            "real_sigma_head": real_head.state_dict(),
            "real_temperature": real_temperature,
        },
        output_dir / "probability_heads.pt",
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
