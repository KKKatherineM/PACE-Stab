"""ProStab-style asymmetric WT-context and ESM-mutation-effect model."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from lora import LoRALinear, inject
from prefix import last_two_forward


class PythiaProjection(nn.Module):
    """The only trainable structure-side module; Pythia p/c states stay frozen."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.LayerNorm(512))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AAPreferenceHead(nn.Module):
    """Exact ProStab MLP ordering, adapted only from 21 classes to P1's 20 AA alphabet."""
    def __init__(self):
        super().__init__()
        dropout = 0.2  # `MLPConfig.dropout` in the actual ProStab source.
        self.net = nn.Sequential(
            nn.Linear(1792, 1024), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(1024, 1024), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(1024, 1024), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(1024, 20),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ESMDeltaHead(nn.Module):
    """The explicit ProStab mutation-effect scalar branch."""
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 640), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(640, 320), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(320, 160), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(160, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class ScalarFusionHead(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class ProStabStylePythiaFusion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        import esm

        config = resolve_paths(config)
        self.config = config
        self.esm, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(foundation_path('esm2')))
        for parameter in self.esm.parameters():
            parameter.requires_grad = False
        self.insertions = inject(self.esm, config["lora"])
        self.prefix_layer = config["esm"]["prefix_last_layer"]
        dropout = float(config["training"]["dropout"])
        self.pythia_projection = PythiaProjection()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=1792, nhead=8, dim_feedforward=3584, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.wt_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.aa_preference_head = AAPreferenceHead()
        self.esm_delta_head = ESMDeltaHead(dropout)
        self.final_mlp = ScalarFusionHead(dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen ESM stays in eval mode; LoRA dropout remains train/eval with task mode.
        self.esm.eval()
        for module in self.esm.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode)
        return self

    def esm_final(self, wt: list[torch.Tensor], mut: list[torch.Tensor]):
        device = next(self.esm.parameters()).device
        values = wt + mut
        lengths = torch.tensor([value.size(0) for value in values], device=device)
        padded = pad_sequence(values, batch_first=True).to(device, dtype=next(self.esm.parameters()).dtype, non_blocking=True)
        final = last_two_forward(self.esm, padded, lengths, self.prefix_layer)
        return final[:len(wt)], final[len(wt):], lengths[:len(wt)]

    def forward(self, wt, mut, struct_unique, position0, wt_aa_index, mut_aa_index, wt_inverse_index, return_shapes: bool = False):
        hwt_unique, hmut, wt_lengths = self.esm_final(wt, mut)
        inverse = wt_inverse_index.to(hmut.device, non_blocking=True)
        unique_residue_lengths = wt_lengths - 2
        max_residues = int(unique_residue_lengths.max())
        wt_residue_unique = hwt_unique[:, 1:max_residues + 1]
        mut_residue = hmut[:, 1:max_residues + 1]
        struct_hidden_unique = struct_unique.to(hmut.device, dtype=hmut.dtype, non_blocking=True)
        if struct_hidden_unique.shape[:2] != wt_residue_unique.shape[:2] or struct_hidden_unique.shape[-1] != 256:
            raise ValueError(f"unique ESM/Pythia padded shape mismatch: {tuple(wt_residue_unique.shape)} vs {tuple(struct_hidden_unique.shape)}")
        positions = position0.to(hmut.device, non_blocking=True)
        residue_lengths = unique_residue_lengths[inverse]
        if (positions >= residue_lengths).any():
            raise ValueError("mutation position lies outside aligned residue-only tensor")
        padding_mask_unique = torch.arange(max_residues, device=hmut.device)[None] >= unique_residue_lengths[:, None]

        projected_struct = self.pythia_projection(struct_hidden_unique)
        wt_concat = torch.cat((wt_residue_unique, projected_struct), dim=-1)
        wt_joint_unique = self.wt_transformer(wt_concat, src_key_padding_mask=padding_mask_unique)
        batch_index = torch.arange(hmut.size(0), device=hmut.device)
        site_embedding = wt_joint_unique[inverse, positions]
        aa_scores = self.aa_preference_head(site_embedding)
        wt_score = aa_scores.gather(1, wt_aa_index.to(hmut.device).unsqueeze(1)).squeeze(1)
        mut_score = aa_scores.gather(1, mut_aa_index.to(hmut.device).unsqueeze(1)).squeeze(1)
        # ProStab's code uses mutant preference minus WT preference.
        delta_context = mut_score - wt_score

        wt_residue = wt_residue_unique[inverse]
        esm_site_delta = mut_residue[batch_index, positions] - wt_residue[batch_index, positions]
        delta_seq = self.esm_delta_head(esm_site_delta)
        final_input = torch.stack((delta_context, delta_seq), dim=-1)
        prediction = self.final_mlp(final_input)
        if return_shapes:
            return prediction, {
                "hwt_unique": list(wt_residue_unique.shape), "Mut_ESM": list(mut_residue.shape),
                "struct_unique": list(struct_hidden_unique.shape), "Projected_Pythia": list(projected_struct.shape),
                "WT_concat": list(wt_concat.shape), "wt_joint_unique": list(wt_joint_unique.shape),
                "wt_inverse_index": list(inverse.shape),
                "site_embedding": list(site_embedding.shape), "AA_scores": list(aa_scores.shape),
                "delta_context": list(delta_context.shape), "ESM_site_delta": list(esm_site_delta.shape),
                "delta_seq": list(delta_seq.shape), "final_input": list(final_input.shape),
                "prediction": list(prediction.shape), "padding_mask_unique": list(padding_mask_unique.shape),
            }
        return prediction

    def parameter_report(self) -> dict:
        esm_base = [(name, parameter) for name, parameter in self.esm.named_parameters() if "lora_" not in name]
        lora = [(name, parameter) for name, parameter in self.esm.named_parameters() if "lora_" in name]
        modules = {
            "ESM base": esm_base, "ESM LoRA": lora, "Pythia backbone": [],
            "PythiaProjection": list(self.pythia_projection.named_parameters()),
            "WT Transformer": list(self.wt_transformer.named_parameters()),
            "20-AA predictor": list(self.aa_preference_head.named_parameters()),
            "ESM delta head": list(self.esm_delta_head.named_parameters()),
            "Final MLP": list(self.final_mlp.named_parameters()),
        }
        return {name: {"total": sum(parameter.numel() for _, parameter in values), "trainable": sum(parameter.numel() for _, parameter in values if parameter.requires_grad)} for name, values in modules.items()}
