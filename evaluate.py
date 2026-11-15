#!/usr/bin/env python3
"""Score predictions against separately supplied labels by observation_uid."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def score(predictions, labels, column='M6'):
    for frame in (predictions, labels):
        if frame.empty or frame.observation_uid.isna().any() or (not frame.observation_uid.is_unique):
            raise ValueError('IDs must be unique and nonempty')
    if set(predictions.observation_uid) != set(labels.observation_uid):
        raise ValueError('Prediction and label populations differ')
    merged = predictions[['observation_uid', column]].merge(labels[['observation_uid', 'target_ddg']], on='observation_uid', validate='one_to_one')
    (p, y) = (merged[column].to_numpy(float), merged.target_ddg.to_numpy(float))
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Predictions and targets must be finite')
    variable = len(y) > 1 and np.std(p) > 0 and (np.std(y) > 0)
    return {'n': len(y), 'spearman': float(spearmanr(p, y)[0]) if variable else None, 'pearson': float(pearsonr(p, y)[0]) if variable else None, 'rmse': float(np.sqrt(np.mean((p - y) ** 2))), 'mae': float(np.mean(np.abs(p - y)))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--column', default='M6')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(pd.read_csv(args.predictions, dtype={'observation_uid': str}), pd.read_csv(args.labels, dtype={'observation_uid': str}), args.column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result))
if __name__ == '__main__':
    main()
