"""Thin frozen wrapper around the existing Pythia structure encoder.

This module deliberately imports, rather than copies, Pythia graph/AMPNN code.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

STRUCTURE_ENCODER_ROOT = Path(__file__).resolve().parents[1]
if str(STRUCTURE_ENCODER_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURE_ENCODER_ROOT))

from models.pythia_encoder import PythiaStructureEncoder  # noqa: E402

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}


class PythiaCompatibilityModel(PythiaStructureEncoder):
    """Expose the frozen residue states and the original 20 canonical AA logits."""

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str | torch.device = "cpu") -> "PythiaCompatibilityModel":
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pythia checkpoint not found: {checkpoint}")
        model = cls()
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    @torch.inference_mode()
    def encode_with_logits(
        self, node_features: torch.Tensor, edge_features: torch.Tensor, original_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([L, 128] states, [L, 20] canonical raw logits)``.

        No softmax, probability ratio, DDG score, or parameter update is used.
        """
        states = self.forward(node_features, edge_features)[:original_length]
        all_logits = self.lm_heads(states)
        if all_logits.shape != (original_length, 21):
            raise RuntimeError(f"Unexpected Pythia head shape: {tuple(all_logits.shape)}")
        return states, all_logits[:, : len(AA_ORDER)]
