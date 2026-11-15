"""Cached ESM prefixes plus metadata-keyed frozen p+c site features."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import hashlib, json, os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

COUNTS = {"train": 190427, "val": 25909, "test": 28172, "s669": 669}
KEYS = ["protein_id", "mutation", "position0"]
def key(sequence: str) -> str: return hashlib.sha256(sequence.encode()).hexdigest()
def canonical(split: str, paths: dict) -> pd.DataFrame:
    if split == "s669":
        x = pd.read_csv(paths["s669_csv"])
        return pd.DataFrame({"sample_id":x.source_row.astype(str), "protein_id":x.protein_id.astype(str), "mutation":x.mutation.astype(str), "position0":x.position0.astype(int), "target":x.target_ddg.astype(float), "wt_seq":x.wt_sequence, "mut_seq":x.mut_sequence})
    x = pd.read_csv(manifest_file(paths["manifests_root"], split))
    x = x.rename(columns={"mutation_position":"position0", "target_ddg":"target"})
    x.sample_id = x.sample_id.astype(str)
    return x[["sample_id","protein_id","mutation","position0","target","wt_seq","mut_seq"]].copy()
def feature_root(split: str, paths: dict) -> Path:
    return Path(paths["pythia_feature_root"])/("s669" if split == "s669" else f"megascale_by_split/{split}")
def aligned_frame(split: str, paths: dict) -> tuple[pd.DataFrame, Path]:
    base = canonical(split, paths); root = feature_root(split, paths)
    index_name = "pythia_pc_s669_features_index.csv" if split == "s669" else "pythia_pc_features_index.csv"
    p = pd.read_csv(root/index_name)
    for frame in (base, p):
        frame.protein_id=frame.protein_id.astype(str); frame.mutation=frame.mutation.astype(str); frame.position0=frame.position0.astype(int)
    # Mapping is key based, never feature-row based.  source_row/sample_id and split are audited too.
    p = p.rename(columns={"feature_row":"pythia_row", "source_row":"sample_id", "target_ddg":"pythia_target"})
    p.sample_id = p.sample_id.astype(str)
    merged = base.merge(p[["pythia_row","sample_id","split",*KEYS,"pythia_target"]], on=["sample_id",*KEYS], validate="one_to_one")
    expected_split = "s669" if split == "s669" else split
    if len(merged) != COUNTS[split] or len(base) != len(p) or not (merged["split"] == expected_split).all() or not np.allclose(merged.target, merged.pythia_target, atol=1e-6):
        raise ValueError(f"{split}: sample_id/protein/mutation/position/label/split alignment failure")
    if merged.duplicated(["sample_id",*KEYS]).any(): raise ValueError(f"{split}: duplicate aligned key")
    # Same cached-v2 sampler length: cached prefix states include BOS/EOS.
    merged["sequence_length"] = merged.wt_seq.str.len().astype(int) + 2
    # Same cached-v2 WT-aware grouping keys; cache lookup uses the identical SHA-256.
    merged["wt_key"] = [key(sequence) for sequence in merged.wt_seq]
    merged["mut_key"] = [key(sequence) for sequence in merged.mut_seq]
    return merged.reset_index(drop=True), root

class PackedStore:
    """Unchanged packed/sharded cache access: disk, bounded rolling RAM, or full RAM."""
    def __init__(self, root: Path, mode="disk", max_shards=8, prefetch=True):
        self.root=Path(root); self.index=resolve_paths(json.loads((self.root/"index.json").read_text())); self.mode=mode; self.max_shards=max_shards; self.loaded=OrderedDict(); self.prefetch=prefetch and mode=="rolling_ram"; self.pool=ThreadPoolExecutor(max_workers=1) if self.prefetch else None; self.pending={}; self.loads=0
        if mode == "ram": self.preload_all()
    def _load(self, shard): return torch.load(self.root/shard,map_location="cpu")
    def _insert(self, shard, value):
        self.loaded[shard]=value; self.loaded.move_to_end(shard)
        if self.mode == "rolling_ram":
            while len(self.loaded)>self.max_shards: self.loaded.popitem(last=False)
    def preload_all(self):
        for shard in sorted({e["shard"] for e in self.index.values()}): self._insert(shard,self._load(shard)); self.loads+=1
    def prefetch_shards(self, shards):
        if self.pool:
            for shard in shards:
                if shard not in self.loaded and shard not in self.pending: self.pending[shard]=self.pool.submit(self._load,shard)
    def get(self, cache_key):
        entry=self.index[cache_key]; shard=entry["shard"]
        if shard in self.loaded: self.loaded.move_to_end(shard); value=self.loaded[shard]
        else:
            f=self.pending.pop(shard,None); value=f.result() if f else self._load(shard); self.loads+=1; self._insert(shard,value)
        return value["hidden"][entry["start"]:entry["stop"]]
def linux_memory_info():
    """Physical-RAM diagnostics; MemAvailable includes reclaimable page cache."""
    wanted={'MemTotal','MemAvailable','MemFree','Cached','Buffers','SwapTotal','SwapFree'}
    try:
        values={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            name,value,*_=line.replace(':','').split()
            if name in wanted: values[name]=int(value)*1024
        if 'MemAvailable' in values:
            return values,'/proc/meminfo MemAvailable'
    except OSError: pass
    return {'MemAvailable':os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')},'FALLBACK'
def get_linux_mem_available_bytes(): return linux_memory_info()[0]['MemAvailable']
def fullram_peak_bytes(cache_root, splits=('train','val')):
    """Actual simultaneous PackedStore payloads: one WT store per Dataset."""
    root=Path(cache_root); items=[]
    for split in splits:
        for role,path in (('wt',root/'wt'),('mutant',root/'mut'/split)):
            size=sum(x.stat().st_size for x in path.glob('shard_*.pt'))
            items.append({'split':split,'role':role,'bytes':size})
    return items,sum(x['bytes'] for x in items)
class CachedMutationDataset(Dataset):
    def __init__(self, split, paths, cache_root, data_cfg=None):
        cfg=data_cfg or {}; self.frame,self.feature_root=aligned_frame(split,paths); self.split=split
        mode=cfg.get("cache_mode","disk"); roots=[Path(cache_root)/"wt",Path(cache_root)/"mut"/split]
        needed=sum(x.stat().st_size for root in roots for x in root.glob("shard_*.pt"))
        if mode=="ram" and needed>get_linux_mem_available_bytes()*(1-float(cfg.get("ram_safety_fraction",.2))): raise RuntimeError("RAM cache safety margin would be violated")
        self.wt=PackedStore(roots[0],mode,int(cfg.get("rolling_ram_shards",8))); self.mut=PackedStore(roots[1],mode,int(cfg.get("rolling_ram_shards",8)))
        names=("z_struct_site_p.npy","z_struct_site_c.npy","delta_logit_pc.npy","wt_aa_onehot.npy","mut_aa_onehot.npy")
        self.site_p,self.site_c,self.delta_pc,self.wt_aa,self.mut_aa=(np.load(self.feature_root/n,mmap_mode="r") for n in names)
        expected=((self.site_p,128),(self.site_c,128),(self.delta_pc,1),(self.wt_aa,20),(self.mut_aa,20))
        if any(x.shape != (COUNTS[split],width) for x,width in expected): raise ValueError(f"{split}: p+c feature dimensionality mismatch")
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]; row=int(r.pythia_row); get=lambda x:torch.from_numpy(np.asarray(x[row],dtype=np.float32).copy())
        return {"wt":self.wt.get(key(r.wt_seq)),"mut":self.mut.get(key(r.mut_seq)),"wt_key":key(r.wt_seq),"position0":int(r.position0),"target":float(r.target),"site_p":get(self.site_p),"site_c":get(self.site_c),"delta_pc":get(self.delta_pc),"wt_aa":get(self.wt_aa),"mut_aa":get(self.mut_aa),"item":i,"length":len(r.wt_seq)+2}
def collate(items, deduplicate_wt=True):
    wt_indices={}; unique=[]; inverse=[]
    for n,x in enumerate(items):
        k=x["wt_key"] if deduplicate_wt else x["wt_key"]+str(n)
        if k not in wt_indices: wt_indices[k]=len(unique); unique.append(x["wt"])
        inverse.append(wt_indices[k])
    lengths=torch.tensor([x["length"] for x in items]); padded=int(lengths.max())*len(items)
    stack=lambda n:torch.stack([x[n] for x in items])
    return {"wt":unique,"mut":[x["mut"] for x in items],"wt_inverse_index":torch.tensor(inverse),"position0":torch.tensor([x["position0"] for x in items]),"target":torch.tensor([x["target"] for x in items]),"site_p":stack("site_p"),"site_c":stack("site_c"),"delta_pc":stack("delta_pc"),"wt_aa":stack("wt_aa"),"mut_aa":stack("mut_aa"),"item":torch.tensor([x["item"] for x in items]),"length":lengths,"unique_wt_count":len(unique),"last2_sequence_count":len(unique)+len(items),"padding_ratio":padded/int(lengths.sum())}
