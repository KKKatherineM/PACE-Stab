#!/usr/bin/env python3
"""Download official foundation weights to their configured locations and verify SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['esm', 'pythia', 'all'], default='all')
    parser.add_argument('--list', action='store_true', help='Print download URLs without downloading')
    args = parser.parse_args()
    for item in json.loads((ROOT / 'configs/pretrained.json').read_text())['files']:
        if args.only != 'all' and (not item['name'].startswith(args.only)):
            continue
        if args.list:
            print(f"{item['path']}\n  {item['url']}")
            continue
        target = ROOT / item['path']
        if target.exists():
            if sha256(target) != item['sha256']:
                raise RuntimeError(f'Existing file has an unexpected hash: {target}')
            print(f"Verified {item['name']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.download')
        if temporary.exists():
            raise FileExistsError(f'Remove or resume the incomplete download first: {temporary}')
        try:
            print(f"Downloading {item['name']}", flush=True)
            with urllib.request.urlopen(item['url'], timeout=60) as source, temporary.open('xb') as out:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            if sha256(temporary) != item['sha256']:
                raise RuntimeError(f"Unexpected upstream hash: {item['name']}; original file not replaced")
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
if __name__ == '__main__':
    main()
