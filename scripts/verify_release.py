#!/usr/bin/env python3
"""Verify the packaged file inventory without dependencies, GPU or network."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    failures = []
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            failures.append(row['path'])
    if failures:
        raise SystemExit('Changed or missing release files: ' + ', '.join(failures))
    print(json.dumps({'status': 'PASS', 'files': len(manifest['files']),
                      'inference': False, 'network': False}, indent=2))


if __name__ == '__main__':
    main()
