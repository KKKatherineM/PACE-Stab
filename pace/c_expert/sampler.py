"""Epoch-random WT-aware, length-bucketed mutation batches."""
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
        self.frame, self.batch_size, self.seed, self.bucket_width, self.epoch = frame, batch_size, seed, bucket_width, 0
        self.groups = defaultdict(lambda: defaultdict(list))
        minimum = int(frame.sequence_length.min())
        for index, row in frame[["wt_key", "sequence_length"]].iterrows():
            self.groups[(int(row.sequence_length) - minimum) // bucket_width][row.wt_key].append(index)
        self.last_audit = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng, batches = np.random.default_rng(self.seed + self.epoch), []
        for bucket in rng.permutation(list(self.groups)):
            groups = list(self.groups[bucket].items())
            rng.shuffle(groups)
            for _, indices in groups:
                values = np.asarray(indices, dtype=np.int64).copy()
                rng.shuffle(values)
                batches.extend(values[start:start + self.batch_size].tolist() for start in range(0, len(values), self.batch_size))
        rng.shuffle(batches)
        seen = [index for batch in batches for index in batch]
        self.last_audit = {"epoch_sample_count": len(seen), "unique_sample_count": len(set(seen)), "duplicate_count": len(seen) - len(set(seen)), "missing_count": len(self.frame) - len(set(seen))}
        if self.last_audit["duplicate_count"] or self.last_audit["missing_count"]:
            raise RuntimeError(f"sampler coverage failure: {self.last_audit}")
        yield from batches

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for groups in self.groups.values() for indices in groups.values())
