"""Small artifact utilities for reproducible diagnostic experiments."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inventory(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_inventory(directory, expected):
    for name, checksum in expected.items():
        if sha256(Path(directory) / name) != checksum:
            raise RuntimeError(f'Artifact changed: {directory}/{name}')
