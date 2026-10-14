"""Gated clean baseline only; deliberately exposes no noise/sparsity execution flags."""
import argparse
import concurrent.futures
import multiprocessing
import shutil
import hashlib
import json
import platform
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from utils.physics_informed_burgers import (FEATURES,NOTEBOOKS,dataset,hashes,write_json,validate,train_seed,summarize)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100000,help='Other horizons are smoke tests, never a validated baseline.')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--reuse-completed',type=Path,help='Copy completed seed artifacts to a fresh destination after checking the scientific configuration.')
    args=parser.parse_args()
    if not 1<=args.steps<=100000: parser.error('steps must be 1..100000')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    protected=hashes(NOTEBOOKS)
    sources=hashes((*NOTEBOOKS,'utils/physics_informed_burgers.py','runs/run_physics_informed_burgers.py',
        'prog/mlps.py','prog/featlib.py','utils/derivative_utils.py','Datasets/data/processed/burg_gen/burg_gen.py'))
    config=dict(dataset=dict(seed=0,nu=.02,T=.1,N=256,dt=.002,observations=640,time_samples=10,space_samples=64),
        seeds=[0,1,2,3,4,5],seed_0_gate=True,repeat_seeds=[1,2,3,4,5],steps=args.steps,
        architecture=dict(hidden_layers=3,width=64,first_omega=1,hidden_omega=1),
        physical_head='xi[0]*u*ux + xi[1]*uxx',initial_coefficients=[0.,0.],coefficient_initialization='constant zeros independent of truth',
        surrogate_initialization='fresh SirenMLP immediately after torch.manual_seed(seed)',batch_seed='10000 + initialization_seed',
        features=FEATURES,normalization=False,physical_coordinates=True,lr=.0005,optimizer='Adam',
        adam=dict(betas=[.9,.999],eps=1e-8,weight_decay=0),lambda_data=1,lambda_pde=1,l1=0,
        batch_size=640,batch_sampling='with replacement',dtype='float32',device='cpu',threads=1,
        recovery=dict(loose_abs=[.25,.02],strong_abs=[.10,.01],strict=True,targets=[-1,.02],order=['transport','diffusion']),
        gate='Mechanical/numerical checks pass; full 100k seed-0 endpoint strong, field relative L2 <0.05; finite metrics and independent residual agreement.',
        deviations=['Explicit two-coefficient head replaces SymNet as requested.',
          'Fresh seed-controlled SIREN skips RNG consumption from discarded notebook models/pretraining; not an exact notebook trajectory replay.',
          'Independent batch RNG; CPU used because CUDA unavailable.',
          'Ground-truth dataset seed fixed at 0 for all optimization replicates.'],
        noise_protocol=dict(levels=[0,.01,.02,.05,.10,.20],executed=False,formula='u_clean + sigma * std(original 640 clean observations, ddof=0) * epsilon',noise_seed_separate=True),
        sampling_protocol=dict(ratios=[1,.75,.5,.25,.1],counts=[640,480,320,160,64],denominator=640,executed=False,
          method='nested uniform random subsets without replacement from a saved permutation; separate sampling seed'),
        workers=args.workers,reuse_completed=str(args.reuse_completed) if args.reuse_completed else None,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,source_hashes=sources,smoke=args.steps!=100000)
    write_json(args.output/'config.json',config)
    (args.output/'reproduction.txt').write_text(' '.join([sys.executable,'runs/run_physics_informed_burgers.py','--output',str(args.output),'--steps',str(args.steps)]+(['--validate-only'] if args.validate_only else [])+['--workers',str(args.workers)]+(['--reuse-completed',str(args.reuse_completed)] if args.reuse_completed else []))+'\nUse a new output directory; existing destinations are rejected.\n')
    _,_,ref,grid=dataset()
    np.savez_compressed(args.output/'reference.npz',**grid,**{f'ref_{n}':ref[n]['physical'] for n in (*FEATURES,'u_t')})
    checks=validate();write_json(args.output/'validation.json',checks)
    print('Mechanical and numerical validation passed',flush=True)
    if args.validate_only:return
    summaries=[]
    if args.reuse_completed:
        prior=json.loads((args.reuse_completed/'config.json').read_text())
        scientific_keys=('dataset','steps','architecture','physical_head','initial_coefficients',
            'coefficient_initialization','surrogate_initialization','batch_seed','features','normalization',
            'physical_coordinates','lr','optimizer','adam','lambda_data','lambda_pde','l1','batch_size',
            'batch_sampling','dtype','device','threads','recovery')
        # JSON round-trip normalizes tuple/list representation before comparison.
        normalized=json.loads(json.dumps(config))
        for key in scientific_keys:
            if prior[key]!=normalized[key]: raise ValueError(f'Cannot reuse changed configuration: {key}')
        prior_validation=json.loads((args.reuse_completed/'validation.json').read_text())
        if not prior_validation.get('seed_0_gate_passed'):
            raise ValueError('Reuse requires a validated full seed-0 gate')
        copied_hashes={}
        for seed in config['seeds']:
            old=args.reuse_completed/f'seed_{seed}'
            if not (old/'summary.json').exists(): continue
            summary=json.loads((old/'summary.json').read_text())
            if summary['step']!=args.steps: raise ValueError('Incomplete seed summary')
            target=args.output/old.name
            shutil.copytree(old,target)
            for file in old.rglob('*'):
                if file.is_file():
                    digest=hashlib.sha256(file.read_bytes()).hexdigest()
                    assert hashlib.sha256((target/file.relative_to(old)).read_bytes()).hexdigest()==digest
                    copied_hashes[str(file.relative_to(args.reuse_completed))]=digest
            summaries.append(summary)
        write_json(args.output/'reused_artifacts.json',dict(source=str(args.reuse_completed),hashes=copied_hashes,
            scientific_configuration_identical=True,source_config=prior))
    if not any(s['seed']==0 for s in summaries):
        summaries.append(train_seed(args.output/'seed_0',0,args.steps))
    s=next(s for s in summaries if s['seed']==0)
    gate=args.steps==100000 and s['strong'] and s['u_rel_l2']<.05
    checks.update(seed_0_gate_passed=gate,notebooks_unchanged=hashes(NOTEBOOKS)==protected)
    write_json(args.output/'validation.json',checks)
    summarize(args.output,summaries)
    if gate:
        pending=[seed for seed in range(1,6) if not any(s['seed']==seed for s in summaries)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(train_seed,args.output/f'seed_{seed}',seed,args.steps):seed for seed in pending}
            for job in concurrent.futures.as_completed(jobs):
                summaries.append(job.result())
                summaries.sort(key=lambda s:s['seed'])
                summarize(args.output,summaries)
    checks.update(notebooks_unchanged=hashes(NOTEBOOKS)==protected,completed_seeds=[s['seed'] for s in summaries],
        all_endpoints_strong=all(s['strong'] for s in summaries),all_independent_residual_checks_passed=True)
    assert checks['notebooks_unchanged']
    write_json(args.output/'validation.json',checks)


if __name__=='__main__': main()
