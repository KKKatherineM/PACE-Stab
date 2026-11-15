"""Minimal, audited LoRA insertion for the final two frozen ESM2 layers."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

from dataclasses import dataclass
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base, self.scale, self.dropout = base, alpha / rank, nn.Dropout(dropout)
        for parameter in base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** .5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + ((self.dropout(value) @ self.lora_a.t()) @ self.lora_b.t()) * self.scale


@dataclass(frozen=True)
class Insertion:
    module_name: str
    layer: int
    target: str


def inject(model: nn.Module, config: dict) -> list[Insertion]:
    targets = {"query": "q_proj", "value": "v_proj"}
    requested = {targets[item] for item in config["target_modules"]}
    selected = set(range(len(model.layers) - config["last_n_layers"], len(model.layers)))
    found = []
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "layers" and parts[1].isdigit() and int(parts[1]) in selected and parts[-1] in requested and isinstance(module, nn.Linear):
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(module, config["rank"], config["alpha"], config["dropout"]))
            found.append(Insertion(name, int(parts[1]), parts[-1]))
    if len(found) != len(selected) * len(requested):
        raise RuntimeError(f"audited q/v insertion failed: {found}")
    return found
