"""Exact fair-esm prefix/resume split at the audited layer boundary."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import torch

def prefix_forward(esm,tokens,prefix_last_layer):
    pad=tokens.eq(esm.padding_idx); x=esm.embed_scale*esm.embed_tokens(tokens)
    if esm.token_dropout:
        x.masked_fill_((tokens==esm.mask_idx).unsqueeze(-1),0.0); ratio=(tokens==esm.mask_idx).sum(-1).to(x.dtype)/(~pad).sum(-1); x=x*(1-.12)/(1-ratio)[:,None,None]
    x=x*(1-pad.unsqueeze(-1).type_as(x)); x=x.transpose(0,1); mask=None if not pad.any() else pad
    for index in range(prefix_last_layer): x,_=esm.layers[index](x,self_attn_padding_mask=mask,need_head_weights=False)
    return x.transpose(0,1)

def last_two_forward(esm,prefix_hidden,lengths,prefix_last_layer):
    max_len=prefix_hidden.shape[1]; mask=torch.arange(max_len,device=prefix_hidden.device)[None]>=lengths[:,None]; x=prefix_hidden.transpose(0,1); attention_mask=None if not mask.any() else mask
    for index in range(prefix_last_layer,len(esm.layers)): x,_=esm.layers[index](x,self_attn_padding_mask=attention_mask,need_head_weights=False)
    return esm.emb_layer_norm_after(x).transpose(0,1)
