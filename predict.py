#!/usr/bin/env python3
"""Predict one expert or the fixed six-member PACE ensemble from generated features."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from pace.inputs import read_population, aligned_cache_index
from pace.ensemble.combine import MEMBERS, combine
from pace_release import RELEASE_ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--member', choices=['all', *MEMBERS], default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.member == 'all':
        frames = {}
        with tempfile.TemporaryDirectory(prefix='pace-members-', dir=args.output.parent) as tmp:
            for name in MEMBERS:
                path = Path(tmp) / f'{name}.csv'
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--cache', str(args.cache.resolve()), '--output', str(path.resolve()), '--member', name, '--device', args.device, '--batch-size', str(args.batch_size)], check=True)
                frames[name] = pd.read_csv(path, dtype={'observation_uid': str})
            combine(frames).to_csv(args.output, index=False)
        return
    import torch
    from pace.ensemble import member
    torch.set_num_threads(4)
    frame = read_population(args.cache / 'population.csv')
    index = aligned_cache_index(frame, args.cache / 'pythia/pythia_cache_index.csv')
    member.ESM = args.cache / 'esm'
    member.PYTHIA = args.cache / 'pythia'
    manifest = json.loads((RELEASE_ROOT / 'checkpoints/checkpoints_manifest.json').read_text())
    checkpoint = RELEASE_ROOT / next((x['path'] for x in manifest['members'] if x['name'] == args.member))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'PACE checkpoint missing: {checkpoint}; see README.md')
    device = torch.device(args.device)
    values = member.scratch(frame, checkpoint, device, args.batch_size) if args.member.startswith('P') else member.context(frame, index, checkpoint, device, args.batch_size)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError('Invalid model predictions')
    output = frame[['observation_uid', 'protein_id', 'position0']].copy()
    output[args.member] = values
    output.to_csv(args.output, index=False)
if __name__ == '__main__':
    main()
