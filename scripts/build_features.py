#!/usr/bin/env python3
"""Build features locally; generated arrays are not part of the source release."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace.inputs import read_population
from pace.features import build

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['esm', 'pythia', 'both'], default='both')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        parser.error('Batch and shard sizes must be positive')
    frame = read_population(args.input)
    build.torch.set_num_threads(4)
    build.OUT = args.output.resolve()
    (build.OUT / 'reports').mkdir(parents=True, exist_ok=True)
    population = build.OUT / 'population.csv'
    if population.exists():
        previous = read_population(population)
        if not previous.equals(frame):
            raise ValueError('Refusing to reuse a cache directory for different inputs')
    else:
        frame.to_csv(population, index=False)
    report = {}
    if args.stage in ('esm', 'both'):
        report['esm'] = build.esm_build(frame, args.device, args.batch_size, args.shard_size)
    if args.stage in ('pythia', 'both'):
        report['pythia'] = build.pythia_build(frame, args.device)
    (build.OUT / 'reports' / f'{args.stage}.json').write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
