"""ESM2 last-2 LoRA + scratch evo/site-local + frozen Pythia p+c fusion."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from lora import LoRALinear, inject
from prefix import last_two_forward
from models.residue_evo_encoder import ResidueEvoEncoder

class PythiaPcSiteAdaptedEncoder(nn.Module):
    """Verbatim p+c pre-regression representation from esm_pythia_pc_site_fusion/model.py."""
    def __init__(self, dropout: float, projection_dim: int=64):
        super().__init__()
        self.site_pc_fusion=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(dropout))
        self.site_projection=nn.Linear(128,projection_dim)
        self.compat_projection=nn.Linear(1,projection_dim)
        self.gate=nn.Linear(projection_dim*2,2)
    def forward(self,site_p,site_c,delta_pc,wt_onehot,mut_onehot):
        h_site=self.site_projection(self.site_pc_fusion(torch.cat((site_p,site_c),-1)))
        h_comp=self.compat_projection(delta_pc)
        gates=torch.softmax(self.gate(torch.cat((h_site,h_comp),-1)),dim=-1)
        z_struct=gates[:,:1]*h_site+gates[:,1:2]*h_comp
        return torch.cat((z_struct,wt_onehot,mut_onehot),dim=-1)
class Head(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(256,512),nn.ReLU(),nn.Dropout(d),nn.Linear(512,128),nn.ReLU(),nn.Dropout(d),nn.Linear(128,1))
    def forward(self,x): return self.net(x).squeeze(-1)
class ScratchEvoLoRAPcFusion(nn.Module):
    def __init__(self,cfg,pythia_checkpoint: Path):
        super().__init__(); import esm
        cfg=resolve_paths(cfg); pythia_checkpoint=release_path(pythia_checkpoint); self.cfg=cfg; self.esm,self.alphabet=esm.pretrained.load_model_and_alphabet_local(str(foundation_path('esm2'))); self.esm.eval()
        for p in self.esm.parameters(): p.requires_grad=False
        self.insertions=inject(self.esm,cfg['lora']); self.prefix_layer=cfg['esm']['prefix_last_layer']
        # These are deliberately constructed from scratch for every seed.  No evo
        # or site-gate state is ever loaded from outside this run.
        self.evo_encoder=ResidueEvoEncoder(); self.site_gate=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))
        payload=torch.load(pythia_checkpoint,map_location='cpu'); self.pythia_checkpoint=str(pythia_checkpoint)
        self.pythia=PythiaPcSiteAdaptedEncoder(float(payload['dropout']))
        self.pythia.load_state_dict({k:v for k,v in payload['model_state_dict'].items() if not k.startswith('regressor.')},strict=True)
        for p in self.pythia.parameters(): p.requires_grad=False
        self.evo_projection=nn.Linear(256,256); self.struct_projection=nn.Linear(104,256); self.gate=nn.Linear(512,1); self.regressor=Head(cfg['training']['dropout'])
    def initialize_scratch(self):
        """Record the intentional scratch initialization without loading state."""
        if not any(p.detach().abs().sum().item() > 0 for p in self.evo_encoder.parameters()):
            raise RuntimeError('scratch evo_encoder unexpectedly has an all-zero initialization')
        if not all(p.requires_grad for p in self.evo_encoder.parameters()):
            raise RuntimeError('scratch evo_encoder must be trainable')
        if not all(p.requires_grad for p in self.site_gate.parameters()):
            raise RuntimeError('scratch site_gate must be trainable')
    def train(self,mode=True):
        super().train(mode); self.esm.eval(); self.pythia.eval()
        for m in self.esm.modules():
            if isinstance(m,LoRALinear): m.dropout.train(mode)
        return self
    def final(self,wt,mut):
        device=next(self.esm.parameters()).device; values=wt+mut; lengths=torch.tensor([x.size(0) for x in values],device=device)
        padded=pad_sequence(values,batch_first=True).to(device,dtype=next(self.esm.parameters()).dtype,non_blocking=True)
        final=last_two_forward(self.esm,padded,lengths,self.prefix_layer)
        return final[:len(wt)],final[len(wt):],lengths[:len(wt)]
    def forward(self,wt,mut,pos,site_p,site_c,delta_pc,wt_aa,mut_aa,wt_inverse_index=None,return_all=False):
        hwt_unique,hmut,wt_lengths=self.final(wt,mut)
        if wt_inverse_index is None: wt_inverse_index=torch.arange(len(mut),device=hmut.device)
        else: wt_inverse_index=wt_inverse_index.to(hmut.device,non_blocking=True)
        hwt=hwt_unique[wt_inverse_index]; lengths=wt_lengths[wt_inverse_index]; z=[]
        for j,p in enumerate(pos.tolist()):
            delta=hmut[j,1:lengths[j]-1]-hwt[j,1:lengths[j]-1]; residues=self.evo_encoder(delta); site=residues[p]; local=residues[max(0,p-10):min(len(residues),p+11)].mean(0); g=torch.sigmoid(self.site_gate(torch.cat((site,local)))); z.append(g*site+(1-g)*local)
        z=torch.stack(z)
        with torch.no_grad(): struct=self.pythia(site_p,site_c,delta_pc,wt_aa,mut_aa)
        he=self.evo_projection(z); hs=self.struct_projection(struct); gate=torch.sigmoid(self.gate(torch.cat((he,hs),-1))); pred=self.regressor(gate*he+(1-gate)*hs)
        return (pred,gate.squeeze(-1),z,struct,hwt,hmut) if return_all else pred
    def report(self):
        total=sum(p.numel() for p in self.parameters()); train=sum(p.numel() for p in self.parameters() if p.requires_grad)
        modules={'LoRA':[(n,p) for n,p in self.esm.named_parameters() if 'lora_' in n], 'evo_encoder':list(self.evo_encoder.named_parameters()), 'site_gate':list(self.site_gate.named_parameters()), 'evo_projection':list(self.evo_projection.named_parameters()), 'struct_projection':list(self.struct_projection.named_parameters()), 'fusion_gate':list(self.gate.named_parameters()), 'prediction_head':list(self.regressor.named_parameters()), 'ESM backbone':[(n,p) for n,p in self.esm.named_parameters() if 'lora_' not in n], 'Pythia':list(self.pythia.named_parameters())}
        counts={name:{'total':sum(p.numel() for _,p in values),'trainable':sum(p.numel() for _,p in values if p.requires_grad),'frozen':sum(p.numel() for _,p in values if not p.requires_grad)} for name,values in modules.items()}
        return {'total_parameters':total,'trainable_parameters':train,'frozen_parameters':total-train,'modules':counts,'pythia_checkpoint':self.pythia_checkpoint,'evo_initialization':'random_scratch'}
