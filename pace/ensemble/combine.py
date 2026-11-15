"""The fixed PACE six-member mean, joined by observation identity."""
import numpy as np
import pandas as pd
MEMBERS = ('P42', 'P2024', 'P3407', 'C42', 'C2024', 'C3407')

def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if set(frames) != set(MEMBERS):
        raise ValueError('Exactly the six formal members are required')
    result = frames[MEMBERS[0]][['observation_uid']].copy()
    if result.empty or result.observation_uid.isna().any() or (not result.observation_uid.is_unique):
        raise ValueError('Prediction IDs must be nonempty and unique')
    ids = set(result.observation_uid)
    for name in MEMBERS:
        part = frames[name]
        if not part.observation_uid.is_unique or set(part.observation_uid) != ids:
            raise ValueError(f'{name}: prediction population differs')
        if not np.isfinite(part[name].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: nonfinite predictions')
        result = result.merge(part[['observation_uid', name]], on='observation_uid', validate='one_to_one', sort=False)
    result['mu_P'] = result[list(MEMBERS[:3])].mean(axis=1)
    result['mu_C'] = result[list(MEMBERS[3:])].mean(axis=1)
    result['M6'] = (result.mu_P + result.mu_C) / 2
    return result
