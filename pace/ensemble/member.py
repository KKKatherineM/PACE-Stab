"""Frozen single-member prediction with the formal P/C equations."""
from __future__ import annotations
import hashlib
import importlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import yaml
from pace_release import RELEASE_ROOT, resolve_paths
EXPERT = RELEASE_ROOT / 'pace'
ESM = Path('cache/input/esm')
PYTHIA = Path('cache/input/pythia')
AA = 'ACDEFGHIKLMNPQRSTVWY'

class Store:

    def __init__(self, root):
        self.root = root
        self.index = resolve_paths(json.loads((root / 'index.json').read_text()))
        self.loaded = OrderedDict()

    def get(self, s):
        key = hashlib.sha256(s.encode()).hexdigest()
        e = self.index[key]
        shard = e['shard']
        if shard not in self.loaded:
            self.loaded[shard] = torch.load(self.root / shard, map_location='cpu')
        return self.loaded[shard]['hidden'][e['start']:e['stop']]

def clear():
    for n in list(sys.modules):
        if n in ('model', 'wrapper', 'lora', 'prefix', 'data', 'models') or n.startswith('models.'):
            sys.modules.pop(n, None)

def scratch(x, checkpoint, dev, bs):
    d = EXPERT / 'p_expert'
    sys.path.insert(0, str(d))
    clear()
    model_mod = importlib.import_module('model')
    cfg = resolve_paths(yaml.safe_load((d / 'config.yaml').read_text()))
    p = torch.load(checkpoint, map_location='cpu')
    m = model_mod.ScratchEvoLoRAPcFusion(cfg, Path(cfg['pythia_checkpoints'][p['seed']]))
    inc = m.load_state_dict(p['model_state_dict'], strict=True)
    if inc.missing_keys or inc.unexpected_keys:
        raise RuntimeError('FINAL_EXPERT_INFERENCE_FAIL')
    m.to(dev).eval()
    arr = {n: np.load(PYTHIA / f'{n}.npy', mmap_mode='r') for n in ('z_struct_site_p', 'z_struct_site_c', 'delta_logit_pc', 'wt_aa_onehot', 'mut_aa_onehot')}
    (wt, mut) = (Store(ESM / 'wt'), Store(ESM / 'mut'))
    out = np.empty(len(x), np.float32)
    with torch.inference_mode():
        for (_, g) in x.groupby('protein_id', sort=False):
            state = wt.get(str(g.wt_sequence.iloc[0]))
            rows = g.index.to_numpy()
            for st in range(0, len(rows), bs):
                ch = rows[st:st + bs]
                z = x.iloc[ch]
                b = {'wt': [state], 'mut': [mut.get(str(s)) for s in z.mut_sequence], 'position0': torch.tensor(z.position0.to_numpy(), dtype=torch.long), 'site_p': torch.from_numpy(np.asarray(arr['z_struct_site_p'][ch])).float(), 'site_c': torch.from_numpy(np.asarray(arr['z_struct_site_c'][ch])).float(), 'delta_pc': torch.from_numpy(np.asarray(arr['delta_logit_pc'][ch])).float(), 'wt_aa': torch.from_numpy(np.asarray(arr['wt_aa_onehot'][ch])).float(), 'mut_aa': torch.from_numpy(np.asarray(arr['mut_aa_onehot'][ch])).float(), 'wt_inverse_index': torch.zeros(len(ch), dtype=torch.long)}
                b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for (k, v) in b.items()}
                out[ch] = m(b['wt'], b['mut'], b['position0'], b['site_p'], b['site_c'], b['delta_pc'], b['wt_aa'], b['mut_aa'], b['wt_inverse_index']).detach().float().cpu().numpy()
    return out

def context(x, ci, checkpoint, dev, bs):
    d = EXPERT / 'c_expert'
    sys.path.insert(0, str(d))
    clear()
    model_mod = importlib.import_module('model')
    p = torch.load(checkpoint, map_location='cpu')
    m = model_mod.ProStabStylePythiaFusion(p['config'])
    inc = m.load_state_dict(p['model_state_dict'], strict=True)
    if inc.missing_keys or inc.unexpected_keys:
        raise RuntimeError('FINAL_EXPERT_INFERENCE_FAIL')
    m.to(dev).eval()
    [setattr(q, 'requires_grad', False) for q in m.parameters()]
    keys = ci.pythia_cache_key.to_numpy()
    (wt, mut) = (Store(ESM / 'wt'), Store(ESM / 'mut'))
    out = np.empty(len(x), np.float32)
    with torch.inference_mode():
        for (pid, g) in x.groupby('protein_id', sort=False):
            rows = g.index.to_numpy()
            key = str(keys[rows[0]])
            if not np.all(keys[rows] == key):
                raise RuntimeError('FINAL_SCORING_ALIGNMENT_FAIL')
            grid = np.load(PYTHIA / 'fullgrid' / f'{key}.npz')
            pc = torch.from_numpy(grid['pc_state_full'].copy()).unsqueeze(0)
            mask = torch.from_numpy(grid['structure_mask'].copy()).unsqueeze(0)
            state = wt.get(str(g.wt_sequence.iloc[0]))
            for st in range(0, len(rows), bs):
                ch = rows[st:st + bs]
                z = x.iloc[ch]
                (hwt, hmut, lens) = m.esm_final([state], [mut.get(str(s)) for s in z.mut_sequence])
                R = int((lens - 2).max())
                wr = hwt[:, 1:R + 1]
                mr = hmut[:, 1:R + 1]
                struct = pc.to(device=dev, dtype=hmut.dtype)
                smask = mask[:, :R].to(device=dev, dtype=hmut.dtype)
                if struct.shape[:2] != wr.shape[:2] or smask.shape != struct.shape[:2]:
                    raise RuntimeError('FINAL_EXPERT_INFERENCE_FAIL')
                padding = torch.arange(R, device=dev)[None] >= (lens - 2)[:, None]
                joint = m.wt_transformer(torch.cat((wr, m.pythia_projection(struct) * smask.unsqueeze(-1)), dim=-1), src_key_padding_mask=padding)
                pos = torch.tensor(z.position0.to_numpy(), dtype=torch.long, device=dev)
                inv = torch.zeros(len(ch), dtype=torch.long, device=dev)
                b = torch.arange(len(ch), device=dev)
                aa = m.aa_preference_head(joint[inv, pos])
                wi = torch.tensor([AA.index(a) for a in z.wt_aa], device=dev)
                mi = torch.tensor([AA.index(a) for a in z.mut_aa], device=dev)
                dc = aa.gather(1, mi[:, None]).squeeze(1) - aa.gather(1, wi[:, None]).squeeze(1)
                ds = m.esm_delta_head(mr[b, pos] - wr[inv, pos])
                out[ch] = m.final_mlp(torch.stack((dc, ds), dim=-1)).detach().float().cpu().numpy()
    return out
