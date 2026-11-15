"""Local LoRA implementation for audited fair-esm q/v modules."""
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
    def __init__(self, base, rank, alpha, dropout):
        super().__init__(); self.base=base; self.scale=alpha/rank; self.dropout=nn.Dropout(dropout)
        for p in base.parameters(): p.requires_grad=False
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features)); self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank)); nn.init.kaiming_uniform_(self.lora_a,a=5**.5)
    def forward(self,x): return self.base(x)+((self.dropout(x)@self.lora_a.t())@self.lora_b.t())*self.scale

@dataclass(frozen=True)
class Insertion:
    module_name:str; layer:int; target:str

def inject(model,cfg):
    targets={'query':'q_proj','value':'v_proj'}; requested={targets[x] for x in cfg['target_modules']}; selected=set(range(len(model.layers)-cfg['last_n_layers'],len(model.layers))); found=[]
    for name,module in list(model.named_modules()):
        p=name.split('.')
        if len(p)>=4 and p[0]=='layers' and p[1].isdigit() and int(p[1]) in selected and p[-1] in requested and isinstance(module,nn.Linear):
            parent=model
            for part in p[:-1]: parent=getattr(parent,part)
            setattr(parent,p[-1],LoRALinear(module,cfg['rank'],cfg['alpha'],cfg['dropout'])); found.append(Insertion(name,int(p[1]),p[-1]))
    if len(found)!=len(selected)*len(requested): raise RuntimeError(f'audited q/v insertion failed: {found}')
    return found
