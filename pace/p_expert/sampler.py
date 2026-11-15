"""Epoch-random WT-aware, length-bucketed mutation batch sampler."""
from __future__ import annotations

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file

from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler


class WTAwareLengthBatchSampler(Sampler[list[int]]):
    def __init__(self, frame, batch_size: int, seed: int, bucket_width: int = 4):
        self.frame, self.batch_size, self.seed, self.bucket_width = frame, batch_size, seed, bucket_width
        self.epoch = 0
        self.groups = defaultdict(lambda: defaultdict(list))
        minimum = int(frame.sequence_length.min())
        for i, row in frame[["wt_key", "sequence_length"]].iterrows():
            bucket = (int(row.sequence_length) - minimum) // bucket_width
            self.groups[bucket][row.wt_key].append(i)
        self.last_audit = {}

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for bucket in rng.permutation(list(self.groups)):
            wt_groups = list(self.groups[bucket].items())
            rng.shuffle(wt_groups)
            for _, indices in wt_groups:
                values = np.asarray(indices, dtype=np.int64).copy()
                rng.shuffle(values)
                batches.extend(values[i:i + self.batch_size].tolist() for i in range(0, len(values), self.batch_size))
        # Bucket/group order changes each epoch; retaining individual group chunks maximizes WT reuse.
        rng.shuffle(batches)
        seen = [i for b in batches for i in b]
        unique = len(set(seen))
        self.last_audit = {"epoch_sample_count": len(seen), "unique_sample_count": unique,
                           "duplicate_count": len(seen) - unique, "missing_count": len(self.frame) - unique}
        if self.last_audit["duplicate_count"] or self.last_audit["missing_count"]:
            raise RuntimeError(f"sampler coverage failure: {self.last_audit}")
        yield from batches

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for groups in self.groups.values() for indices in groups.values())
