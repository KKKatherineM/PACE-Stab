"""Inference-only residue encoder adapted minimally from Pythia's AMPNN."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn


class AMPNNLayer(nn.Module):
    """Original Pythia attention message-passing layer (unchanged computation)."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, dropout: float = 0.2, neighbor_num: int = 32):
        super().__init__()
        self.multihead_message = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.feed_forwards_message = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
        self.multihead_update = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.feed_forwards_update = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
        self.message_transitions = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.neighbor_num = neighbor_num

    def forward(self, h0: torch.Tensor, e0: torch.Tensor) -> torch.Tensor:
        h1, _ = self.multihead_message(h0[0].unsqueeze(0).repeat(self.neighbor_num, 1, 1), h0, h0)
        h2 = self.layer_norms[1](self.feed_forwards_message(self.layer_norms[0](h1)) + h0)
        message = self.message_transitions(torch.cat((h2, e0), dim=-1))
        h3, _ = self.multihead_update(message, h2, h2)
        return self.layer_norms[3](self.feed_forwards_update(self.layer_norms[2](h3) + h2))


class PythiaStructureEncoder(nn.Module):
    """Load Pythia weights and expose residue, rather than DDG, representations.

    Inputs have Pythia's original shapes: node_features ``[K, L, 28]`` and
    edge_features ``[K, L, 27]``. ``forward`` returns ``[L, 128]``.
    """

    def __init__(self, embed_dim: int = 128, edge_dim: int = 27, node_dim: int = 28, n_heads: int = 8, layer_nums: int = 3, dropout: float = 0.2, neighbor_num: int = 32):
        super().__init__()
        self.neighbor_num = neighbor_num
        self.init_node_embed = nn.Linear(node_dim, embed_dim, bias=False)
        self.init_edge_embed = nn.Linear(edge_dim, embed_dim, bias=False)
        self.layer_list = nn.ModuleList([AMPNNLayer(embed_dim, n_heads, dropout, neighbor_num) for _ in range(layer_nums)])
        # Retained solely so the original checkpoint loads strictly. It is never used.
        self.lm_heads = nn.Linear(embed_dim, 21)
        self.layer_norms = nn.LayerNorm(embed_dim)

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "PythiaStructureEncoder":
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pythia checkpoint not found: {checkpoint}")
        model = cls()
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        h = self.init_node_embed(node_features)
        edge = self.init_edge_embed(edge_features)
        for layer in self.layer_list:
            h = layer(h, edge)
        # Pythia feeds h.sum(0) to its amino-acid head. Keep that per-residue state.
        return h.sum(dim=0)

    @torch.inference_mode()
    def encode_graph(self, node_features: torch.Tensor, edge_features: torch.Tensor, original_length: int | None = None) -> torch.Tensor:
        embeddings = self(node_features, edge_features)
        return embeddings[:original_length] if original_length is not None else embeddings

    @staticmethod
    def pool_residues(residue_embeddings: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Optional explicit pooling; extraction never calls this method by default."""
        if reduction == "mean":
            return residue_embeddings.mean(dim=0)
        if reduction == "sum":
            return residue_embeddings.sum(dim=0)
        raise ValueError("reduction must be 'mean' or 'sum'")
