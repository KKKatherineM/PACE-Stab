#!/usr/bin/env python3
"""Evaluate either formal checkpoint into the prescribed per-seed directory."""

# Locate the bundled resources when invoked from any working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "pace_release.py").is_file())
_release_sys.path.insert(0, str(_release_root))
from pace_release import RELEASE_ROOT, release_path, resolve_paths, foundation_path, manifest_file
import argparse
from pathlib import Path
import torch,yaml
from model import ScratchEvoLoRAPcFusion
from train import assert_scratch_configuration, evaluate_checkpoint
ROOT=Path(__file__).parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-type',choices=('best','epoch20'),required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();cfg=resolve_paths(yaml.safe_load((ROOT/'config.yaml').read_text()));cfg['seed']=a.seed
 if any(x in str(a.checkpoint) for x in ('residue_evo_mutation_centered_pooling','site_local_gated')): raise RuntimeError('evaluation checkpoint must belong to this scratch-evo run')
 model=ScratchEvoLoRAPcFusion(cfg,Path(cfg['pythia_checkpoints'][a.seed]));model.initialize_scratch();model.to(a.device)
 # Evaluation restores evo/site-gate strictly from the supplied run checkpoint; no external evo load exists.
 evaluate_checkpoint(model,a.checkpoint,cfg,torch.device(a.device),a.checkpoint_type)
if __name__=='__main__':main()
