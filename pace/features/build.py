"""ESM prefix and residue-aligned Pythia features for a target-free population."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pace_release import RELEASE_ROOT, release_path, foundation_path
OUT = Path('cache/input')
PYTHIA_ROOT = RELEASE_ROOT / 'third_party/pythia_structure'
ID = ['dataset', 'observation_uid', 'source_record_index']
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'

def fail(message: str) -> None:
    raise RuntimeError(message)

def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def esm_build(frame: pd.DataFrame, device_name: str, batch_size: int, per_shard: int) -> dict[str, object]:
    import esm
    from pace.p_expert.prefix import prefix_forward
    device = torch.device(device_name)
    (model, alphabet) = esm.pretrained.load_model_and_alphabet_local(str(foundation_path('esm2')))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    converter = alphabet.get_batch_converter()
    root = OUT / 'esm'

    def build_role(role: str, sequences: pd.Series) -> dict[str, object]:
        records = {sha_text(str(sequence)): str(sequence) for sequence in sequences}
        items = sorted(records.items())
        role_root = root / role
        role_root.mkdir(parents=True, exist_ok=False)
        index: dict[str, dict[str, object]] = {}
        shards: list[str] = []
        start_time = time.time()
        for first in range(0, len(items), per_shard):
            group = items[first:first + per_shard]
            packed: list[torch.Tensor] = []
            offset = 0
            for subfirst in range(0, len(group), batch_size):
                sub = group[subfirst:subfirst + batch_size]
                (_, _, tokens) = converter(sub)
                with torch.no_grad():
                    hidden = prefix_forward(model, tokens.to(device), 31).cpu()
                for (row, (key, sequence)) in enumerate(sub):
                    value = hidden[row, :len(sequence) + 2].to(torch.float16).contiguous()
                    if tuple(value.shape) != (len(sequence) + 2, 1280):
                        fail('CACHE_GENERATION_FAIL')
                    packed.append(value)
                    index[key] = {'shard': f'shard_{first // per_shard:05d}.pt', 'start': offset, 'stop': offset + len(value), 'sequence_length': len(sequence)}
                    offset += len(value)
            name = f'shard_{first // per_shard:05d}.pt'
            torch.save({'hidden': torch.cat(packed), 'dtype': 'float16', 'keys': [key for (key, _) in group]}, role_root / name)
            shards.append(name)
            print(json.dumps({'stage': 'esm', 'role': role, 'done': min(first + per_shard, len(items)), 'total': len(items)}), flush=True)
        (role_root / 'index.json').write_text(json.dumps(index, sort_keys=True), encoding='utf-8')
        return {'unique_sequences': len(items), 'shards': shards, 'seconds': time.time() - start_time, 'dtype': 'float16', 'hidden_dimension': 1280}
    report = {'model': 'esm2_t33_650M_UR50D', 'prefix_blocks': '0-30', 'prefix_last_layer': 31, 'state_semantics': 'BASE_ESM_PREFIX_STATE_BLOCK31_INPUT', 'bos_index': 0, 'residue_token_offset': 1, 'eos_index': 'L+1', 'device': str(device), 'wt': build_role('wt', frame.wt_sequence), 'mut': build_role('mut', frame.mut_sequence)}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'cache_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report

def global_map(sequence: str, observed: str) -> dict[int, int]:
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    (aligner.match_score, aligner.mismatch_score) = (2.0, -1.0)
    (aligner.open_gap_score, aligner.extend_gap_score) = (-3.0, -0.5)
    alignment = aligner.align(sequence, observed)[0]
    result: dict[int, int] = {}
    coords = alignment.coordinates
    for segment in range(coords.shape[1] - 1):
        (s0, s1) = (coords[0, segment], coords[0, segment + 1])
        (o0, o1) = (coords[1, segment], coords[1, segment + 1])
        if s1 - s0 == o1 - o0:
            result.update({int(s0 + offset): int(o0 + offset) for offset in range(s1 - s0)})
    return result

def pythia_build(frame: pd.DataFrame, device_name: str) -> dict[str, object]:
    sys.path.insert(0, str(PYTHIA_ROOT))
    sys.path.insert(0, str(PYTHIA_ROOT / 'pythia_mutation_compatibility'))
    from Bio.Data.IUPACData import protein_letters_3to1
    from model_wrapper import PythiaCompatibilityModel
    from pace.features.graph import build_pythia_graph
    aa3 = {key.upper(): value for (key, value) in protein_letters_3to1.items()}
    device = torch.device(device_name)
    p_model = PythiaCompatibilityModel.from_checkpoint(foundation_path('pythia_p'), device)
    c_model = PythiaCompatibilityModel.from_checkpoint(foundation_path('pythia_c'), device)
    p_model.eval()
    c_model.eval()
    root = OUT / 'pythia'
    full_root = root / 'fullgrid'
    full_root.mkdir(parents=True, exist_ok=False)
    n = len(frame)
    arrays = {'z_struct_site_p': np.lib.format.open_memmap(root / 'z_struct_site_p.npy', mode='w+', dtype=np.float32, shape=(n, 128)), 'z_struct_site_c': np.lib.format.open_memmap(root / 'z_struct_site_c.npy', mode='w+', dtype=np.float32, shape=(n, 128)), 'delta_logit_pc': np.lib.format.open_memmap(root / 'delta_logit_pc.npy', mode='w+', dtype=np.float32, shape=(n, 1)), 'wt_aa_onehot': np.lib.format.open_memmap(root / 'wt_aa_onehot.npy', mode='w+', dtype=np.uint8, shape=(n, 20)), 'mut_aa_onehot': np.lib.format.open_memmap(root / 'mut_aa_onehot.npy', mode='w+', dtype=np.uint8, shape=(n, 20))}
    index_rows: list[dict[str, object]] = []
    protein_rows: list[dict[str, object]] = []
    mapping_failures: list[dict[str, object]] = []
    for (count, (protein_id, group)) in enumerate(frame.groupby('protein_id', sort=False), 1):
        first = group.iloc[0]
        consistency = group[['wt_sequence', 'structure_path', 'chain_id']].drop_duplicates()
        if len(consistency) != 1:
            fail('CACHE_MAPPING_REGRESSION_FAIL')
        (nodes, edges, residue_map) = build_pythia_graph(release_path(first.structure_path), chain_id=str(first.chain_id))
        with torch.no_grad():
            (p_state, p_logits) = p_model.encode_with_logits(nodes.to(device), edges.to(device), len(residue_map))
            (c_state, c_logits) = c_model.encode_with_logits(nodes.to(device), edges.to(device), len(residue_map))
        p_raw = p_state.cpu().numpy().astype(np.float32)
        c_raw = c_state.cpu().numpy().astype(np.float32)
        p_logits = p_logits.cpu().numpy().astype(np.float32)
        c_logits = c_logits.cpu().numpy().astype(np.float32)
        observed = ''.join((aa3.get(str(item['resname_3']).upper(), 'X') for item in residue_map))
        mapping = global_map(str(first.wt_sequence), observed)
        length = len(str(first.wt_sequence))
        (p_full, c_full) = (np.zeros((length, 128), np.float32), np.zeros((length, 128), np.float32))
        mask = np.zeros(length, dtype=np.bool_)
        for (pos, aa) in enumerate(str(first.wt_sequence)):
            node_index = mapping.get(pos)
            if node_index is not None and observed[node_index] == aa:
                (p_full[pos], c_full[pos], mask[pos]) = (p_raw[node_index], c_raw[node_index], True)
        failures = []
        for (feature_row, row) in group.iterrows():
            pos = int(row.position0)
            actual_node = mapping.get(pos)
            expected_node = int(row.embedding_position) if 'embedding_position' in row and pd.notna(row.embedding_position) else actual_node
            valid = actual_node is not None and actual_node == expected_node and (0 <= actual_node < len(residue_map)) and (observed[actual_node] == row.wt_aa) and bool(mask[pos])
            if not valid:
                failures.append({'observation_uid': row.observation_uid, 'source_record_index': int(row.source_record_index), 'expected_node': expected_node, 'actual_node': actual_node, 'expected_wt': row.wt_aa, 'actual_wt': '' if actual_node is None else observed[actual_node]})
                continue
            (wi, mi) = (AA_ORDER.index(str(row.wt_aa)), AA_ORDER.index(str(row.mut_aa)))
            arrays['z_struct_site_p'][feature_row] = p_raw[actual_node]
            arrays['z_struct_site_c'][feature_row] = c_raw[actual_node]
            arrays['delta_logit_pc'][feature_row, 0] = p_logits[actual_node, mi] - p_logits[actual_node, wi] + (c_logits[actual_node, mi] - c_logits[actual_node, wi])
            arrays['wt_aa_onehot'][feature_row] = 0
            arrays['wt_aa_onehot'][feature_row, wi] = 1
            arrays['mut_aa_onehot'][feature_row] = 0
            arrays['mut_aa_onehot'][feature_row, mi] = 1
            index_rows.append({'cache_row': feature_row, **{column: row[column] for column in ID}, 'protein_id': protein_id, 'position0': pos, 'embedding_position': actual_node, 'pythia_cache_key': sha_text(f'input:{protein_id}')})
        key = sha_text(f'input:{protein_id}')
        np.savez_compressed(full_root / f'{key}.npz', p_state_raw=p_raw, c_state_raw=c_raw, pc_state_raw=np.concatenate((p_raw, c_raw), axis=-1), p_state_full=p_full, c_state_full=c_full, pc_state_full=np.concatenate((p_full, c_full), axis=-1), structure_mask=mask)
        metadata = {'dataset': 'input', 'protein_id': protein_id, 'cache_key': key, 'structure_path': str(first.structure_path), 'chain_id': str(first.chain_id), 'sequence_length': length, 'pythia_node_count': len(residue_map), 'observed_sequence': observed, 'mapped_fraction': float(mask.mean()), 'mutation_site_failures': failures, 'mapping_method': 'SPURS_STYLE_FULL_BIOLOGICAL_GRID_EXPLICIT_NODE_MAPPING'}
        (full_root / f'{key}.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        protein_rows.append({'protein_id': protein_id, 'cache_key': key, 'sequence_length': length, 'pythia_node_count': len(residue_map), 'mapped_fraction': float(mask.mean()), 'n_observations': len(group), 'mutation_site_failures': len(failures)})
        mapping_failures.extend(failures)
        print(json.dumps({'stage': 'pythia', 'protein': count, 'total': frame.protein_id.nunique(), 'observations': len(group), 'failures': len(failures)}), flush=True)
    for array in arrays.values():
        array.flush()
    if mapping_failures or len(index_rows) != n:
        pd.DataFrame(mapping_failures).to_csv(OUT / 'reports/pythia_mapping_failures.csv', index=False)
        fail('CACHE_MAPPING_REGRESSION_FAIL')
    index = pd.DataFrame(index_rows).sort_values('cache_row', kind='stable')
    if len(index) != n or not index.observation_uid.is_unique or (not index.cache_row.equals(pd.Series(np.arange(n)))):
        fail('CACHE_MAPPING_REGRESSION_FAIL')
    index.to_csv(root / 'pythia_cache_index.csv', index=False)
    pd.DataFrame(protein_rows).to_csv(root / 'fullgrid_protein_index.csv', index=False)
    report = {'dataset': 'input', 'population_n': n, 'p_checkpoint': str(foundation_path('pythia_p')), 'c_checkpoint': str(foundation_path('pythia_c')), 'site_arrays': {name: [n, 20 if 'onehot' in name else 1 if name == 'delta_logit_pc' else 128] for name in arrays}, 'fullgrid_state_semantics': 'PYTHIA_P_C_STRUCTURAL_ENCODER_STATES_SCATTERED_TO_FULL_BIOLOGICAL_GRID', 'dtype': 'float32', 'mapping_pass': n, 'unique_proteins': int(frame.protein_id.nunique()), 'device': str(device)}
    (root / 'cache_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
