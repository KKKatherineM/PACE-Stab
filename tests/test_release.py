import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pace.inputs import read_population, aligned_cache_index
from pace.ensemble.combine import MEMBERS, combine
from pace_release import manifest_file, release_path
from evaluate import score

class ReleaseTests(unittest.TestCase):

    def test_example_and_compressed_manifests(self):
        frame = read_population(ROOT / 'data/inference_example/mutations.csv')
        self.assertEqual(len(frame), 2)
        for path in frame.structure_path:
            self.assertTrue(release_path(path).is_file())
        for split in ['train', 'val', 'test']:
            self.assertTrue(manifest_file(ROOT / 'data/manifests', split).is_file())

    def test_input_rejects_bad_mutation(self):
        frame = pd.read_csv(ROOT / 'data/inference_example/mutations.csv')
        for (column, value) in [('position0', 0.5), ('wt_aa', 'Z'), ('mut_sequence', 'WRONG')]:
            bad = frame.copy()
            bad.loc[0, column] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.csv'
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    read_population(path)

    def test_cache_order_must_match(self):
        frame = read_population(ROOT / 'data/inference_example/mutations.csv')
        index = frame[['dataset', 'observation_uid', 'source_record_index', 'protein_id', 'position0']].copy()
        index['cache_row'] = [0, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index.csv'
            index.to_csv(path, index=False)
            self.assertEqual(len(aligned_cache_index(frame, path)), 2)
            index.iloc[::-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                aligned_cache_index(frame, path)

    def test_ensemble_aligns_ids_and_uses_fixed_mean(self):
        frames = {name: pd.DataFrame({'observation_uid': ['a', 'b'], name: [i, i + 6]}) for (i, name) in enumerate(MEMBERS)}
        frames['C42'] = frames['C42'].iloc[::-1]
        out = combine(frames)
        np.testing.assert_allclose(out.M6, [2.5, 8.5])
        frames['C42'].loc[frames['C42'].index[0], 'C42'] = np.nan
        with self.assertRaises(ValueError):
            combine(frames)

    def test_ensemble_rejects_missing_ids(self):
        frames = {name: pd.DataFrame({'observation_uid': ['a', 'b'], name: [1, 2]}) for name in MEMBERS}
        frames['C3407'] = frames['C3407'].iloc[:1]
        with self.assertRaises(ValueError):
            combine(frames)

    def test_evaluation_requires_identical_population(self):
        p = pd.DataFrame({'observation_uid': ['a', 'b'], 'M6': [1.0, 2.0]})
        y = pd.DataFrame({'observation_uid': ['b', 'a'], 'target_ddg': [2.0, 1.0]})
        self.assertEqual(score(p, y)['rmse'], 0.0)
        with self.assertRaises(ValueError):
            score(p, y.iloc[:1])

    def test_legacy_prior_path_is_portable(self):
        self.assertEqual(release_path('/old/run/pc_site_compat/best_model_seed42.pt'), ROOT / 'checkpoints/training_prior/best_model_seed42.pt')

class TrainingExportTests(unittest.TestCase):

    def test_export_is_readable_by_both_training_loaders(self):
        import hashlib
        import json
        import torch
        from unittest.mock import patch
        from pace.features.training import export_split
        from pace.p_expert import data as p_data
        from pace.c_expert import data as c_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / 'generated'
            output = root / 'cache'
            manifests = root / 'manifests'
            for path in [scratch / 'esm/wt', scratch / 'esm/mut', scratch / 'pythia/fullgrid', manifests]:
                path.mkdir(parents=True)
            canonical = pd.DataFrame({'sample_id': ['0', '1'], 'protein_id': ['p.pdb'] * 2, 'mutation': ['A1C', 'C2D'], 'position0': [0, 1], 'target': [1.0, -1.0], 'wt_seq': ['AC'] * 2, 'mut_seq': ['CC', 'AD']})
            canonical.rename(columns={'position0': 'mutation_position', 'target': 'target_ddg'}).to_csv(manifests / 'val_manifest.csv.gz', index=False)
            for (role, sequences) in [('wt', ['AC']), ('mut', ['CC', 'AD'])]:
                index = {hashlib.sha256(seq.encode()).hexdigest(): {'shard': 'shard_00000.pt', 'start': i * 4, 'stop': (i + 1) * 4} for (i, seq) in enumerate(sequences)}
                torch.save({'hidden': torch.zeros(len(sequences) * 4, 1280)}, scratch / f'esm/{role}/shard_00000.pt')
                (scratch / f'esm/{role}/index.json').write_text(json.dumps(index))
            for (name, width) in [('z_struct_site_p', 128), ('z_struct_site_c', 128), ('delta_logit_pc', 1), ('wt_aa_onehot', 20), ('mut_aa_onehot', 20)]:
                np.save(scratch / f'pythia/{name}.npy', np.zeros((2, width), np.float32))
            pd.DataFrame({'cache_row': [0, 1], 'observation_uid': ['val:0', 'val:1'], 'protein_id': ['p.pdb'] * 2, 'position0': [0, 1], 'pythia_cache_key': ['key'] * 2}).to_csv(scratch / 'pythia/pythia_cache_index.csv', index=False)
            np.savez(scratch / 'pythia/fullgrid/key.npz', p_state_full=np.zeros((2, 128), np.float32), c_state_full=np.zeros((2, 128), np.float32), structure_mask=np.ones(2, bool))
            export_split(scratch, output, 'val', canonical)
            paths = {'manifests_root': str(manifests), 'pythia_feature_root': str(output / 'pythia_site')}
            with patch.dict(p_data.COUNTS, {'val': 2}), patch.dict(c_data.COUNTS, {'val': 2}):
                p = p_data.CachedMutationDataset('val', paths, output / 'esm_prefix', {'cache_mode': 'disk'})
                c = c_data.ProStabStyleDataset('val', paths, str(output / 'esm_prefix'), str(output / 'pythia_residue'), {'cache_mode': 'disk'})
                self.assertEqual(p[1]['target'], -1.0)
                self.assertEqual(c[1]['target'], -1.0)
                self.assertEqual(tuple(c[1]['struct'].shape), (2, 256))
                self.assertEqual(tuple(p[0]['wt'].shape), (4, 1280))
if __name__ == '__main__':
    unittest.main()
