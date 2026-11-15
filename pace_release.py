"""Portable paths for PACE-Stab code, user-generated caches and downloaded weights."""
from pathlib import Path
import os

RELEASE_ROOT = Path(__file__).resolve().parent
PREFIXES = ("cache/", "checkpoints/", "data/", "results/", "third_party/")

def release_path(value):
    path = Path(value)
    # Original PACE checkpoints record the task-specific prior's former location.
    if path.is_absolute() and "/pc_site_compat/" in path.as_posix():
        return RELEASE_ROOT / "checkpoints/training_prior" / path.name
    return path if path.is_absolute() else RELEASE_ROOT / path

def resolve_paths(value):
    if isinstance(value, dict):
        return {key: resolve_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_paths(item) for item in value]
    if isinstance(value, str) and (value.startswith(PREFIXES) or value.startswith("/")):
        return str(release_path(value))
    return value

def foundation_path(name):
    files = {"esm2": "esm/esm2_t33_650M_UR50D.pt", "pythia_p": "pythia/pythia-p.pt", "pythia_c": "pythia/pythia-c.pt"}
    path = Path(os.environ.get("PACE_" + name.upper(), RELEASE_ROOT / "checkpoints" / files[name])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}. Run python scripts/download_pretrained.py; see README.md.")
    return path.resolve()

def manifest_file(root, split):
    path = Path(root) / f"{split}_manifest.csv"
    return path if path.is_file() else path.with_suffix(".csv.gz")
