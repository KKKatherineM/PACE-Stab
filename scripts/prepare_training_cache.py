#!/usr/bin/env python3
"""Rebuild training caches locally from the exact split manifests and PDB structures."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pace_release import RELEASE_ROOT, resolve_paths
from pace.p_expert.data import canonical
from pace.features import build
from pace.features.training import export_split
from pace.inputs import read_population

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--megascale-pdb-dir', type=Path, default=RELEASE_ROOT / 'data/megascale/structures', help='MegaScale PDB directory (extract data/megascale/structures.tar.gz first)')
    parser.add_argument('--s669-pdb-dir', type=Path, default=RELEASE_ROOT / 'data/s669_structures')
    parser.add_argument('--output', type=Path, default=RELEASE_ROOT / 'cache')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test', 's669'], default=['train', 'val', 'test', 's669'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    build.torch.set_num_threads(4)
    config = resolve_paths(yaml.safe_load((RELEASE_ROOT / 'pace/p_expert/config.yaml').read_text()))
    for split in args.splits:
        frame = canonical(split, config['paths'])
        paths = []
        for pid in frame.protein_id:
            pdb = args.s669_pdb_dir / f'{pid[:-1]}.pdb' if split == 's669' else args.megascale_pdb_dir / f'{Path(pid).stem}.pdb'
            if not pdb.is_file():
                raise FileNotFoundError(pdb)
            paths.append(str(pdb.resolve()))
        scratch = args.output / f'_building_{split}'
        scratch.mkdir(parents=True, exist_ok=False)
        (scratch / 'reports').mkdir()
        population = pd.DataFrame({'observation_uid': [f'{split}:{sid}' for sid in frame.sample_id], 'protein_id': frame.protein_id, 'position0': frame.position0, 'wt_aa': frame.mutation.str[0], 'mut_aa': frame.mutation.str[-1], 'wt_sequence': frame.wt_seq, 'mut_sequence': frame.mut_seq, 'structure_path': paths, 'chain_id': 'A'})
        population.to_csv(scratch / 'population.csv', index=False)
        population = read_population(scratch / 'population.csv')
        build.OUT = scratch.resolve()
        build.esm_build(population, args.device, args.batch_size, 128)
        build.pythia_build(population, args.device)
        export_split(scratch, args.output, split, frame)
        shutil.rmtree(scratch)
        print(json.dumps({'split': split, 'rows': len(frame), 'status': 'READY'}), flush=True)
if __name__ == '__main__':
    main()
