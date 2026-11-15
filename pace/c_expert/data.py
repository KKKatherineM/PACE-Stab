"""Validated immutable ESM-prefix and frozen Pythia p/c residue-state reader."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


COUNTS = {"train": 190427, "val": 25909, "test": 28172, "s669": 669}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def canonical_frame(split: str, paths: dict) -> pd.DataFrame:
    if split == "s669":
        frame = pd.read_csv(paths["s669_csv"])
        return pd.DataFrame({"sample_id": frame.source_row.astype(str), "protein_id": frame.protein_id.astype(str), "mutation": frame.mutation.astype(str), "position0": frame.position0.astype(int), "target": frame.target_ddg.astype(float), "wt_seq": frame.wt_sequence, "mut_seq": frame.mut_sequence})
    frame = pd.read_csv(manifest_file(paths["manifests_root"], split)).rename(columns={"mutation_position": "position0", "target_ddg": "target"})
    frame.sample_id = frame.sample_id.astype(str)
    return frame[["sample_id", "protein_id", "mutation", "position0", "target", "wt_seq", "mut_seq"]].copy()


class PackedStore:
    """Same shard cache behavior as the validated early-fusion implementation."""
    def __init__(self, root: Path, mode: str = "disk", max_shards: int = 8):
        self.root, self.index, self.mode, self.max_shards = root, resolve_paths(json.loads((root / "index.json").read_text())), mode, max_shards
        self.loaded = OrderedDict()
        self.prefetch = mode == "rolling_ram"
        self.pool = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        self.pending = {}
        if mode == "ram":
            self.preload_all()

    def _load(self, shard: str):
        return torch.load(self.root / shard, map_location="cpu")

    def _insert(self, shard: str, value) -> None:
        self.loaded[shard] = value
        self.loaded.move_to_end(shard)
        if self.mode == "rolling_ram":
            while len(self.loaded) > self.max_shards:
                self.loaded.popitem(last=False)

    def preload_all(self) -> None:
        for shard in sorted({entry["shard"] for entry in self.index.values()}):
            self._insert(shard, self._load(shard))

    def get(self, key: str) -> torch.Tensor:
        entry, shard = self.index[key], self.index[key]["shard"]
        if shard not in self.loaded:
            future = self.pending.pop(shard, None)
            self._insert(shard, future.result() if future else self._load(shard))
        self.loaded.move_to_end(shard)
        return self.loaded[shard]["hidden"][entry["start"]:entry["stop"]]


class PythiaResidueStore:
    """Frozen, residue-aligned pretrained p/c states; no task checkpoint."""
    def __init__(self, root: Path):
        self.root = root
        self.p = {path.stem: np.load(path, mmap_mode="r") for path in sorted((root / "p").glob("*.npy"))}
        self.c = {path.stem: np.load(path, mmap_mode="r") for path in sorted((root / "c").glob("*.npy"))}
        if not self.p or set(self.p) != set(self.c):
            raise RuntimeError("Pythia p/c residue cache is absent or incomplete")

    def get(self, protein_id: str, sequence_length: int) -> torch.Tensor:
        name = Path(str(protein_id)).stem
        if name not in self.p:
            raise KeyError(f"missing Pythia residue states for {protein_id}")
        p, c = self.p[name], self.c[name]
        if p.shape != c.shape or p.shape != (sequence_length, 128):
            raise ValueError(f"Pythia/sequence alignment failure for {protein_id}: p={p.shape}, c={c.shape}, sequence={sequence_length}")
        return torch.from_numpy(np.concatenate((np.asarray(p, dtype=np.float32), np.asarray(c, dtype=np.float32)), axis=-1).copy())


class ProStabStyleDataset(Dataset):
    def __init__(self, split: str, paths: dict, esm_cache_root: str, pythia_cache_root: str, data_config: dict):
        self.frame, self.split = canonical_frame(split, paths).reset_index(drop=True), split
        if len(self.frame) != COUNTS[split]:
            raise ValueError(f"unexpected {split} count: {len(self.frame)}")
        mode = data_config.get("cache_mode", "disk")
        roots = [Path(esm_cache_root) / "wt", Path(esm_cache_root) / "mut" / split]
        needed = sum(path.stat().st_size for root in roots for path in root.glob("shard_*.pt"))
        if mode == "ram" and needed > available_memory_bytes() * (1 - float(data_config.get("ram_safety_fraction", .2))):
            raise RuntimeError("RAM cache safety margin would be violated")
        self.wt = PackedStore(roots[0], mode, int(data_config.get("rolling_ram_shards", 8)))
        self.mut = PackedStore(roots[1], mode, int(data_config.get("rolling_ram_shards", 8)))
        self.pythia = PythiaResidueStore(Path(pythia_cache_root))
        self.frame["sequence_length"] = self.frame.wt_seq.str.len().astype(int) + 2
        self.frame["wt_key"] = [sequence_key(sequence) for sequence in self.frame.wt_seq]
        self.frame["mut_key"] = [sequence_key(sequence) for sequence in self.frame.mut_seq]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        return {"wt": self.wt.get(row.wt_key), "mut": self.mut.get(row.mut_key), "wt_key": row.wt_key, "struct": self.pythia.get(row.protein_id, len(row.wt_seq)), "position0": int(row.position0), "target": float(row.target), "wt_aa": row.mutation[0], "mut_aa": row.mutation[-1], "length": len(row.wt_seq) + 2}


def available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def fullram_peak_bytes(cache_root: str, splits=("train", "val")) -> int:
    root = Path(cache_root)
    return sum(path.stat().st_size for split in splits for directory in (root / "wt", root / "mut" / split) for path in directory.glob("shard_*.pt"))


def collate(items: list[dict], deduplicate_wt: bool = True) -> dict:
    wt_indices, unique, struct_unique, inverse = {}, [], [], []
    for item_index, item in enumerate(items):
        key = item["wt_key"] if deduplicate_wt else f"{item['wt_key']}:{item_index}"
        if key not in wt_indices:
            wt_indices[key] = len(unique)
            unique.append(item["wt"])
            struct_unique.append(item["struct"])
        inverse.append(wt_indices[key])
    aa_index = lambda key: torch.tensor([AA_ORDER.index(item[key]) for item in items])
    return {"wt": unique, "mut": [item["mut"] for item in items], "wt_inverse_index": torch.tensor(inverse), "struct_unique": pad_sequence(struct_unique, batch_first=True), "position0": torch.tensor([item["position0"] for item in items]), "target": torch.tensor([item["target"] for item in items]), "wt_aa_index": aa_index("wt_aa"), "mut_aa_index": aa_index("mut_aa"), "length": torch.tensor([item["length"] for item in items]), "unique_wt_count": len(unique), "last2_sequence_count": len(unique) + len(items)}
