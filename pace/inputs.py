"""Validate single mutations and bind cached features to observation identifiers."""
from pathlib import Path
import numpy as np
import pandas as pd
AA = 'ACDEFGHIKLMNPQRSTVWY'
COLUMNS = ['observation_uid', 'protein_id', 'position0', 'wt_aa', 'mut_aa', 'wt_sequence', 'mut_sequence', 'structure_path', 'chain_id']

def read_population(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'observation_uid': str, 'protein_id': str, 'chain_id': str})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    selected = COLUMNS + [c for c in ['dataset', 'source_record_index', 'embedding_position'] if c in frame]
    frame = frame[selected].copy().reset_index(drop=True)
    if frame.empty or frame[COLUMNS].isna().any().any() or (not frame.observation_uid.is_unique):
        raise ValueError('Input must contain unique, nonempty observation IDs and complete mutation fields')
    positions = pd.to_numeric(frame.position0)
    if not np.isfinite(positions).all() or not (positions == np.floor(positions)).all():
        raise ValueError('position0 must contain zero-based integers')
    frame.position0 = positions.astype(int)
    for row in frame.itertuples():
        p = row.position0
        if row.wt_aa not in AA or row.mut_aa not in AA or row.wt_aa == row.mut_aa or (not 0 <= p < len(row.wt_sequence)) or (row.wt_sequence[p] != row.wt_aa) or (row.mut_sequence != row.wt_sequence[:p] + row.mut_aa + row.wt_sequence[p + 1:]):
            raise ValueError(f'Invalid single mutation: {row.observation_uid}')
    frame['dataset'] = frame.get('dataset', 'input')
    frame['source_record_index'] = frame.get('source_record_index', np.arange(len(frame)))
    if frame.duplicated(['dataset', 'source_record_index']).any():
        raise ValueError('Duplicate source record identity')
    for (_, group) in frame.groupby('protein_id'):
        if len(group[['wt_sequence', 'structure_path', 'chain_id']].drop_duplicates()) != 1:
            raise ValueError('One protein_id must identify one WT sequence and structure chain')
    return frame

def aligned_cache_index(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    index = pd.read_csv(path, dtype={'observation_uid': str, 'protein_id': str})
    required = ['dataset', 'observation_uid', 'source_record_index', 'protein_id', 'position0']
    if len(index) != len(frame) or not index[required].reset_index(drop=True).equals(frame[required]):
        raise ValueError('Cache identity/order differs from population')
    if not np.array_equal(index.cache_row.to_numpy(), np.arange(len(frame))):
        raise ValueError('Cache rows are not contiguous and aligned')
    return index
