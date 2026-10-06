"""Serialization for the instrumented replay, kept out of run entrypoints."""
import csv
from pathlib import Path


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def finalize(out):
    """Hash final outputs and implementation without including the manifest itself."""
    from utils.diagnostic_io import inventory,sha256,write_json
    out=Path(out);root=Path(__file__).resolve().parents[1]
    files=inventory(out);files.pop('artifact_manifest.json',None)
    sources=[root/'runs/run_phase19b_transition_instrumented.py',root/'runs/PHASE19B_TRANSITION_INSTRUMENTED.md',
             root/'utils/physics.py',root/'utils/transition_update_geometry.py',root/'utils/transition_geometry.py',Path(__file__),
             *sorted((root/'utils').glob('phase19b_instrumented_*.py'))]
    write_json(out/'artifact_manifest.json',{'artifacts_sha256':files,'implementation_sha256':{str(p.relative_to(root)):sha256(p) for p in sources}})
