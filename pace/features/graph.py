"""Pythia graph construction with explicit, lossless residue metadata.

This isolated compatibility copy intentionally preserves the released
builder's PDB ATOM encounter order.  Historical P1 Pythia tensors use this
raw node order, including unusual insertion-code orderings such as ``1X``
before ``1``.  Biological sequence order is represented separately by the
saved ``(chain, resseq, insertion_code) -> tensor_index`` metadata and is
handled only during full-grid scatter.
"""
from __future__ import annotations
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next((p for p in _ReleasePath(__file__).resolve().parents if (p / 'pace_release.py').is_file()))
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import gzip
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as functional
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index

def _parse_structure(pdb_path: Path):
    if pdb_path.suffix in {'.cif', '.mmcif'}:
        return FastMMCIFParser(QUIET=True).get_structure('protein', str(pdb_path))[0]
    if pdb_path.name.endswith('.pdb.gz'):
        with gzip.open(pdb_path, 'rt') as handle:
            return PDBParser(QUIET=True).get_structure('protein', handle)[0]
    return PDBParser(QUIET=True).get_structure('protein', str(pdb_path))[0]

def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    (b, c_vector) = (ca - n, c - ca)
    return -0.58273431 * np.cross(b, c_vector) + 0.56802827 * b - 0.54067466 * c_vector + ca

def _pad(values: torch.Tensor, length: int) -> torch.Tensor:
    padded = torch.zeros((length, *values.shape[1:]), dtype=values.dtype)
    padded[:len(values)] = values
    return padded

def build_pythia_graph(pdb_file: str | Path, neighbor_num: int=32, chain_id: str | None=None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    pdb_path = Path(pdb_file)
    if not pdb_path.is_file():
        raise FileNotFoundError(f'PDB/mmCIF file not found: {pdb_path}')
    structure = _parse_structure(pdb_path)
    structure.atom_to_internal_coordinates()
    arrays = {key: [] for key in ('ca', 'cb', 'c', 'n', 'o', 'seq', 'resseq', 'chain', 'angles')}
    residue_map: list[dict[str, Any]] = []
    chain_ids: dict[str, int] = {}
    for chain in structure.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue
        chain_number = chain_ids.setdefault(chain.id, len(chain_ids))
        for residue in chain.get_residues():
            if residue.id[0] != ' ' or not all((atom in residue for atom in ('N', 'CA', 'C', 'O'))):
                continue
            (n, ca, c, o) = (residue[name].coord for name in ('N', 'CA', 'C', 'O'))
            cb = residue['CB'].coord if 'CB' in residue else _virtual_cb(n, ca, c)
            try:
                token = three_to_index(residue.get_resname())
            except KeyError:
                token = 20
            ric = residue.internal_coord
            angles = [ric.get_angle(name) or 0.0 for name in ('phi', 'psi', 'omg')]
            arrays['ca'].append(ca)
            arrays['cb'].append(cb)
            arrays['c'].append(c)
            arrays['n'].append(n)
            arrays['o'].append(o)
            arrays['seq'].append([token])
            arrays['resseq'].append([residue.id[1]])
            arrays['chain'].append([chain_number])
            arrays['angles'].append(np.concatenate((np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles)))))
            residue_map.append({'embedding_index': len(residue_map), 'chain_id': chain.id, 'resseq': residue.id[1], 'insertion_code': residue.id[2].strip(), 'resname_3': residue.get_resname()})
    original_length = len(residue_map)
    if original_length == 0:
        raise ValueError(f'No standard residues with N/CA/C/O atoms found in {pdb_path}')
    tensors = {key: torch.tensor(np.asarray(value), dtype=torch.float32) for (key, value) in arrays.items()}
    if original_length < neighbor_num:
        tensors = {key: _pad(value, neighbor_num) for (key, value) in tensors.items()}
    length = len(tensors['ca'])
    (ca, cb) = (tensors['ca'].unsqueeze(1), tensors['cb'].unsqueeze(1))
    backbone = torch.cat((ca, cb, tensors['c'].unsqueeze(1), tensors['o'].unsqueeze(1), tensors['n'].unsqueeze(1)), dim=1)
    distances = torch.sqrt(torch.sum((ca - ca.reshape(1, length, 3)) ** 2, dim=-1))
    (_, indices) = torch.topk(distances, neighbor_num, largest=False)
    relative_resseq = torch.clamp(tensors['resseq'].T - tensors['resseq'], min=-32, max=32)
    relative_chain = (tensors['chain'].T - tensors['chain']).int()
    positions = torch.gather(relative_resseq, 1, indices)
    chains = torch.gather(relative_chain, 1, indices)
    tokens = torch.gather(tensors['seq'].T.repeat(length, 1), 1, indices).long()
    tokens[:, 0] = 21
    angles = torch.gather(tensors['angles'].unsqueeze(1).repeat(1, length, 1), 1, indices.unsqueeze(-1).repeat(1, 1, 6))
    pair_distances = torch.sqrt(torch.square(backbone.unsqueeze(0).unsqueeze(2) - backbone.unsqueeze(1).unsqueeze(3)).sum(dim=-1)).reshape(length, length, 25)
    edge_distances = torch.gather(pair_distances, 1, indices.unsqueeze(-1).repeat(1, 1, 25))
    nodes = torch.cat((functional.one_hot(tokens, 22).float(), angles), dim=-1).transpose(0, 1)
    edges = torch.cat((edge_distances, positions.unsqueeze(-1), chains.unsqueeze(-1)), dim=-1).transpose(0, 1)
    return (nodes.float(), edges.float(), residue_map)
