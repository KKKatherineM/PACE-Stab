#!/usr/bin/env python3
"""Train a preregistered ProStab-style Pythia seed replicate."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from data import ProStabStyleDataset, available_memory_bytes, collate, fullram_peak_bytes
from model import ProStabStylePythiaFusion
from sampler import WTAwareLengthBatchSampler

ROOT = Path(__file__).resolve().parent


def set_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def metrics(prediction, target) -> dict:
    return {"spearman": float(spearmanr(target, prediction).statistic), "pearson": float(pearsonr(target, prediction)[0]), "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))), "mae": float(np.mean(np.abs(prediction - target)))}


def make_loader(dataset, split: str, config: dict, seed: int):
    data = config["data"]
    arguments = {"num_workers": int(data["num_workers"]), "pin_memory": bool(data["pin_memory"]), "collate_fn": partial(collate, deduplicate_wt=bool(data["wt_aware_batching"]))}
    if arguments["num_workers"]:
        arguments.update(persistent_workers=bool(data["persistent_workers"]), prefetch_factor=int(data["prefetch_factor"]))
    if split == "train" and data["wt_aware_batching"]:
        sampler = WTAwareLengthBatchSampler(dataset.frame, int(config["training"]["batch_size"]), seed, int(data["length_bucket_width"]))
        return DataLoader(dataset, batch_sampler=sampler, **arguments), sampler
    return DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, **arguments), None


def call(model, batch, device, return_shapes: bool = False):
    return model(batch["wt"], batch["mut"], batch["struct_unique"], batch["position0"].to(device), batch["wt_aa_index"].to(device), batch["mut_aa_index"].to(device), batch["wt_inverse_index"].to(device), return_shapes)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); predictions, targets = [], []
    for batch in loader:
        value = call(model, batch, device)
        predictions.extend(value.float().cpu().tolist()); targets.extend(batch["target"].tolist())
    return np.asarray(predictions), np.asarray(targets)


def make_optimizer(model, training: dict):
    named = dict(model.named_parameters())
    groups = [
        ("ESM LoRA", "lora_", training["lora_learning_rate"]),
        ("PythiaProjection", "pythia_projection", training["pythia_projection_learning_rate"]),
        ("WT Transformer", "wt_transformer", training["wt_transformer_learning_rate"]),
        ("20-AA predictor", "aa_preference_head", training["aa_predictor_learning_rate"]),
        ("ESM delta head", "esm_delta_head", training["esm_delta_learning_rate"]),
        ("Final MLP", "final_mlp", training["final_mlp_learning_rate"]),
    ]
    params = [{"name": label, "params": [value for key, value in named.items() if value.requires_grad and marker in key], "lr": float(rate)} for label, marker, rate in groups]
    if any(not group["params"] for group in params):
        raise RuntimeError("empty required optimizer parameter group")
    return torch.optim.Adam(params, weight_decay=float(training["weight_decay"]))


def assert_configuration(model, optimizer):
    report = model.parameter_report()
    if report["ESM base"]["trainable"] or report["Pythia backbone"]["trainable"]:
        raise RuntimeError("a frozen backbone has trainable parameters")
    required = ["ESM LoRA", "PythiaProjection", "WT Transformer", "20-AA predictor", "ESM delta head", "Final MLP"]
    if any(report[name]["trainable"] == 0 for name in required):
        raise RuntimeError("a required scratch module is not trainable")
    ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and id(parameter) not in ids]
    if missing:
        raise RuntimeError(f"trainable parameters missing from optimizer: {missing[:3]}")
    return report


def sanity(model, batch, device, optimizer):
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction, shapes = call(model, batch, device, True)
    loss = functional.mse_loss(prediction, batch["target"].to(device))
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise FloatingPointError("non-finite forward")
    loss.backward()
    markers = ["lora_", "pythia_projection", "wt_transformer", "aa_preference_head", "esm_delta_head", "final_mlp"]
    gradients = {marker: any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for name, parameter in model.named_parameters() if marker in name) for marker in markers}
    esm_base_grad = any(parameter.grad is not None for name, parameter in model.esm.named_parameters() if "lora_" not in name)
    if not all(gradients.values()) or esm_base_grad:
        raise RuntimeError(f"sanity gradient failure: {gradients}, esm_base={esm_base_grad}")
    optimizer.zero_grad(set_to_none=True)
    return {"prediction_finite": True, "loss_finite": True, "gradients": gradients, "pythia_backbone_grad": None, "esm_base_grad": None, "shapes": shapes}


def save_checkpoint(path, model, optimizer, epoch, val, best_epoch, config):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_epoch": best_epoch, "validation_metrics": val, "selection_metric": "validation_spearman", "seed": config["seed"], "config": config, "parameter_report": model.parameter_report(), "checkpoint_sources": {"esm_foundation": "esm2_t33_650M_UR50D", "pythia_pc_prior": "frozen residue cache", "megascale": "NONE", "dms_lora": "NONE", "prostab": "NONE", "z_evo": "NONE", "z_struct": "NONE", "fusion": "NONE", "regression_head": "NONE"}}, path)


def evaluate(model, checkpoint, config, device, split: str, run: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    pythia_root = config["cache"]["pythia_root"] if split != "s669" else str(Path(config["cache"]["pythia_root"]) / "s669")
    dataset = ProStabStyleDataset(split, config["paths"], config["cache"]["esm_root"], pythia_root, config["data"])
    loader, _ = make_loader(dataset, split, config, int(config["seed"]))
    prediction, target = predict(model, loader, device)
    result = metrics(prediction, target)
    name = "s669" if split == "s669" else f"megascale_{split}"
    (run / "results" / f"{name}_results.json").write_text(json.dumps({"dataset": split, "checkpoint": str(checkpoint), "best_epoch": payload["best_epoch"], **result}, indent=2) + "\n")
    frame = dataset.frame.copy(); frame["prediction"] = prediction
    frame.to_csv(run / "predictions" / f"{name}_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--run-name", default=None, help="fresh output leaf; does not permit overwrite")
    args = parser.parse_args()
    if args.seed not in (42, 2024, 3407, 123, 456):
        raise ValueError("authorized ProStab-style seeds are 42, 2024, 3407, 123, and 456")
    config = resolve_paths(yaml.safe_load(args.config.read_text())); config["seed"] = args.seed
    if args.startup_only:
        config["data"]["cache_mode"] = "disk"
    set_seed(args.seed); device = torch.device(args.device)
    default_run = f"seed{args.seed}_startup" if args.startup_only else f"seed{args.seed}"
    run = ROOT / "runs" / (args.run_name or default_run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"refusing to overwrite {run}")
    for directory in (run / "checkpoints", run / "logs", run / "predictions", run / "results", run / "config"):
        directory.mkdir(parents=True, exist_ok=True)
    (run / "config" / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if config["data"]["cache_mode"] == "ram":
        required = fullram_peak_bytes(config["cache"]["esm_root"])
        safe = available_memory_bytes() * (1 - float(config["data"].get("ram_safety_fraction", .2)))
        if required > safe:
            raise RuntimeError(f"combined train/val full-RAM cache safety margin would be violated: {required} > {safe}")
    datasets = {split: ProStabStyleDataset(split, config["paths"], config["cache"]["esm_root"], config["cache"]["pythia_root"], config["data"]) for split in ("train", "val")}
    loaders, samplers = {}, {}
    for split, dataset in datasets.items():
        loaders[split], samplers[split] = make_loader(dataset, split, config, args.seed)
    model = ProStabStylePythiaFusion(config).to(device)
    optimizer = make_optimizer(model, config["training"])
    report = assert_configuration(model, optimizer)
    first_batch = next(iter(loaders["train"]))
    sanity_report = sanity(model, first_batch, device, optimizer)
    (run / "logs" / "startup_sanity.json").write_text(json.dumps({"parameter_report": report, "sanity": sanity_report}, indent=2) + "\n")
    if args.startup_only:
        return
    best, best_epoch, stale = -np.inf, 0, 0
    best_path = run / "checkpoints" / f"best_model_seed{args.seed}.pt"
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        samplers["train"].set_epoch(epoch); model.train(); losses, started = [], time.time(); optimizer.zero_grad(set_to_none=True)
        for batch in loaders["train"]:
            prediction = call(model, batch, device)
            loss = functional.mse_loss(prediction, batch["target"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss))
        prediction, target = predict(model, loaders["val"], device); val = metrics(prediction, target)
        improved = val["spearman"] > best
        if improved:
            best, best_epoch, stale = val["spearman"], epoch, 0
            save_checkpoint(best_path, model, optimizer, epoch, val, best_epoch, config)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{key}": value for key, value in val.items()}, "best_epoch": best_epoch, "best_val_spearman": best, "stale": stale, "seconds": time.time() - started, "sampler_audit": samplers["train"].last_audit}
        with (run / "logs" / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch >= int(config["training"]["minimum_epochs"]) and stale >= int(config["training"]["patience"]):
            break
    val_result = evaluate(model, best_path, config, device, "val", run)
    test_result = evaluate(model, best_path, config, device, "test", run)
    s669_result = evaluate(model, best_path, config, device, "s669", run)
    (run / "results" / "complete.json").write_text(json.dumps({"best_epoch": best_epoch, "validation": val_result, "megascale_test": test_result, "s669": s669_result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
