"""Residue-level encoder for ESM WT-to-mutant hidden-state differences."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

import torch
from torch import nn


class ResidueEvoEncoder(nn.Module):
    """Fuse point and unpadded local ESM perturbations into 256-dim residues."""

    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, window: int = 10) -> None:
        super().__init__()
        if window < 0:
            raise ValueError("window must be non-negative")
        self.window = window
        self.point_projection = nn.Linear(input_dim, hidden_dim)
        self.local_projection = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _local_mean(self, delta_hidden: torch.Tensor) -> torch.Tensor:
        """Return inclusive window means without zero-padding, in O(L)."""
        length = delta_hidden.size(0)
        prefix = torch.cat((delta_hidden.new_zeros((1, delta_hidden.size(1))), delta_hidden.cumsum(0)), dim=0)
        positions = torch.arange(length, device=delta_hidden.device)
        starts = (positions - self.window).clamp_min(0)
        ends = (positions + self.window + 1).clamp_max(length)
        return (prefix[ends] - prefix[starts]) / (ends - starts).to(delta_hidden.dtype).unsqueeze(-1)

    def forward(self, delta_hidden: torch.Tensor) -> torch.Tensor:
        if delta_hidden.ndim != 2 or delta_hidden.size(-1) != self.point_projection.in_features:
            raise ValueError(f"Expected delta_hidden [L,{self.point_projection.in_features}], got {tuple(delta_hidden.shape)}")
        point = self.point_projection(delta_hidden)
        local = self.local_projection(self._local_mean(delta_hidden))
        gate = torch.sigmoid(self.gate(torch.cat((point, local), dim=-1)))
        return self.norm(gate * point + (1.0 - gate) * local)
