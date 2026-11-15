"""Export generated split features to the formal training loaders' layout."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

def export_split(scratch: Path, output: Path, split: str, canonical: pd.DataFrame):
    ci = pd.read_csv(scratch / 'pythia/pythia_cache_index.csv', dtype={'protein_id': str, 'observation_uid': str})
    expected = [f'{split}:{sid}' for sid in canonical.sample_id]
    if ci.observation_uid.tolist() != expected or ci.protein_id.tolist() != canonical.protein_id.astype(str).tolist() or (not np.array_equal(ci.position0.to_numpy(), canonical.position0.to_numpy())) or (not np.array_equal(ci.cache_row.to_numpy(), np.arange(len(canonical)))):
        raise ValueError('Training feature identity/order differs from canonical manifest')
    site = output / 'pythia_site' / ('s669' if split == 's669' else f'megascale_by_split/{split}')
    mutant = output / 'esm_prefix/mut' / split
    if site.exists() or mutant.exists():
        raise FileExistsError(f'Training cache already exists for {split}')
    site.mkdir(parents=True)
    mutant.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(scratch / 'esm/mut'), mutant)
    wt = output / 'esm_prefix/wt'
    wt.mkdir(parents=True, exist_ok=True)
    index_path = wt / 'index.json'
    merged = json.loads(index_path.read_text()) if index_path.exists() else {}
    index = json.loads((scratch / 'esm/wt/index.json').read_text())
    for source in (scratch / 'esm/wt').glob('shard_*.pt'):
        name = f"shard_{split}_{source.name.removeprefix('shard_')}"
        if (wt / name).exists():
            raise FileExistsError(wt / name)
        shutil.move(str(source), wt / name)
    for (key, entry) in index.items():
        entry['shard'] = f"shard_{split}_{entry['shard'].removeprefix('shard_')}"
        merged.setdefault(key, entry)
    index_path.write_text(json.dumps(merged, sort_keys=True))
    for source in (scratch / 'pythia').glob('*.npy'):
        shutil.move(str(source), site / source.name)
    ci = pd.read_csv(scratch / 'pythia/pythia_cache_index.csv')
    if not np.array_equal(ci.cache_row.to_numpy(), np.arange(len(canonical))):
        raise ValueError('Training feature rows are not aligned')
    features = pd.DataFrame({'feature_row': np.arange(len(canonical)), 'source_row': canonical.sample_id.astype(str), 'split': split, 'protein_id': canonical.protein_id, 'mutation': canonical.mutation, 'position0': canonical.position0, 'target_ddg': canonical.target})
    features.to_csv(site / ('pythia_pc_s669_features_index.csv' if split == 's669' else 'pythia_pc_features_index.csv'), index=False)
    residues = output / 'pythia_residue'
    if split == 's669':
        residues = residues / 's669'
    for role in ['p', 'c']:
        (residues / role).mkdir(parents=True, exist_ok=True)
    for (pid, group) in ci.groupby('protein_id', sort=False):
        grid = np.load(scratch / 'pythia/fullgrid' / f'{group.pythia_cache_key.iloc[0]}.npz')
        if not grid['structure_mask'].all():
            raise ValueError(f'Training requires a fully aligned structure: {pid}')
        for role in ['p', 'c']:
            value = grid[f'{role}_state_full']
            path = residues / role / f'{Path(pid).stem}.npy'
            if path.exists() and (not np.array_equal(np.load(path), value)):
                raise ValueError(f'Conflicting structural features: {path}')
            if not path.exists():
                np.save(path, value)
