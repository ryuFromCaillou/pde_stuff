"""Read-only scientific audit, then reproduce six-seed plots and seal the artifact manifest."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from utils.burgers_six_seed import SIGMAS, SEEDS, sha, validate_condition, publish
from utils.physics_informed_burgers import write_json

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);r=a.output
    config=json.loads((r/'config.json').read_text())
    for name,digest in config['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Original source changed: {name}')
    for entry in json.loads((r/'reused_artifacts.json').read_text()):
        for name,digest in entry['files'].items():
            if sha(Path(entry['source'])/name)!=digest:raise ValueError('Original artifact changed')
            if sha(r/'raw'/f"sigma_{entry['sigma']:g}"/f"seed_{entry['seed']}"/name)!=digest:raise ValueError('Reused copy changed')
    checked=[];noise_by_seed={};indices=None;clean=None
    for sigma in SIGMAS:
        for seed in SEEDS:
            folder=r/'raw'/f'sigma_{sigma:g}'/f'seed_{seed}'
            h,m=validate_condition(folder,sigma,seed)
            with np.load(folder/'observations.npz') as obs:
                if seed in noise_by_seed:np.testing.assert_array_equal(noise_by_seed[seed],obs['epsilon'])
                noise_by_seed[seed]=obs['epsilon'].copy()
                if indices is not None:
                    np.testing.assert_array_equal(indices,obs['indices'])
                    np.testing.assert_array_equal(clean,obs['clean_targets'])
                indices=obs['indices'].copy();clean=obs['clean_targets'].copy()
            checked.append({'sigma':sigma,'seed':seed,'rows':len(h),'checkpoints':len(m),'passed':True})
    # Compare rerun clean endpoints/crossings against the surviving historical summary.
    old=pd.read_csv(ROOT/'run_results/physics_informed_burgers/clean_baseline_completed/seed_summary.csv',float_precision='round_trip')
    clean_comparison=[]
    for seed in range(2,6):
        s=json.loads((r/'raw'/'sigma_0'/f'seed_{seed}'/'summary.json').read_text())
        row=old[old.seed==seed].iloc[0]
        for name in ('transport','diffusion','transport_abs_error','diffusion_abs_error'):
            np.testing.assert_allclose(s[name],row[name],rtol=0,atol=0)
        for name in ('first_loose_step','first_strong_step'):
            expected=None if pd.isna(row[name]) else int(row[name])
            if s[name]!=expected:raise ValueError(f'Historical clean recovery differs: seed {seed}, {name}')
        clean_comparison.append({'seed':seed,'endpoint_and_first_crossings_exact':True,
            'limitation':'Historical dense histories unavailable; full-trajectory equality cannot be established.'})
    write_json(r/'final_audit.json',dict(passed=True,conditions=checked,original_artifacts_unchanged=True,
        training_sources_unchanged=True,identical_clean_observations=True,noise_paired_across_sigma=True,
        clean_summary_comparison=clean_comparison))
    publish(r,require_all=True)
    # Independent mean check against direct per-seed arithmetic; no reliance on aggregation helper.
    for sigma in SIGMAS:
        for kind,filename in [('history','history.csv'),('checkpoints','checkpoint_metrics.json')]:
            frames=[]
            for seed in SEEDS:
                f=r/'raw'/f'sigma_{sigma:g}'/f'seed_{seed}'/filename
                frames.append(pd.read_csv(f,float_precision='round_trip') if kind=='history' else pd.DataFrame(json.loads(f.read_text())))
            saved=pd.read_csv(r/'averaged'/f'sigma_{sigma:g}'/f'{kind}_mean.csv',float_precision='round_trip')
            for col in saved:
                if col=='step':np.testing.assert_array_equal(saved[col],frames[0][col]);continue
                expected=np.add.reduce([x[col].to_numpy(dtype=float) for x in frames])/6
                np.testing.assert_array_equal(saved[col],expected)
    final=json.loads((r/'final_audit.json').read_text());final['independent_saved_mean_check']=True
    write_json(r/'final_audit.json',final)
    manifest={str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in r.rglob('*')
        if p.is_file() and p.name not in ('manifest.json','execution.log','execution.pid','finalization.log','finalization.pid')}
    write_json(r/'manifest.json',manifest)
    print('All 36 conditions and saved means passed; original artifacts unchanged.',flush=True)

if __name__=='__main__':main()
