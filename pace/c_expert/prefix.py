"""The validated ESM prefix-cache resume path (layers 31--32 online)."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

import torch


def last_two_forward(esm, prefix_hidden: torch.Tensor, lengths: torch.Tensor, prefix_last_layer: int) -> torch.Tensor:
    max_length = prefix_hidden.shape[1]
    padding_mask = torch.arange(max_length, device=prefix_hidden.device)[None] >= lengths[:, None]
    hidden = prefix_hidden.transpose(0, 1)
    attention_mask = None if not padding_mask.any() else padding_mask
    for index in range(prefix_last_layer, len(esm.layers)):
        hidden, _ = esm.layers[index](hidden, self_attn_padding_mask=attention_mask, need_head_weights=False)
    return esm.emb_layer_norm_after(hidden).transpose(0, 1)
