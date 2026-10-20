"""Extend the unchanged Burgers noise experiment; publish only validated six-seed means."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from utils.physics_informed_burgers import write_json
from utils.physics_informed_burgers_noise import SIGMAS, train_condition
from utils.burgers_six_seed import SEEDS, STEPS, sha, validate_condition

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('run_results/physics_informed_burgers/noise_robustness_resume2'))
    p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    a=p.parse_args(); torch.set_num_threads(1)
    if a.plot_only:
        from utils.burgers_six_seed import publish
        publish(a.output,require_all=True); return
    if not a.output.exists():
        source_config=json.loads((a.source/'config.json').read_text())
        for path,digest in source_config['source_hashes'].items():
            if sha(ROOT/path)!=digest: raise ValueError(f'Original implementation mismatch: {path}')
        if source_config['steps']!=STEPS: raise ValueError('Training budget mismatch')
        a.output.mkdir(parents=True)
        config={**source_config,'seeds':list(SEEDS),'workers':a.workers,
                'noise':{**source_config['noise'],'noise_seed_by_initialization':{str(s):700000+s for s in SEEDS}},
                'authoritative_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source':str(a.source.resolve()),'aggregation':'Arithmetic mean at identical optimizer steps; six seeds required; population std ddof=0',
                'noise_exception':'Original noise realization varies with initialization seed (700000+seed), paired across sigma. Preserved to extend the original design.',
                'recovery':{'targets':[-1,.02],'loose_abs':[.25,.02],'strong_abs':[.10,.01],'strict':True,'first_crossing_not_sustained':True},
                'missing_clean_histories':'Clean baseline seeds 2–5 summaries exist but raw histories unavailable; rerun all four.'}
        write_json(a.output/'config.json',config)
        (a.output/'reproduction.txt').write_text(
            f'{sys.executable} runs/run_burgers_six_seed.py --output {a.output} --workers {a.workers}\n'
            f'{sys.executable} runs/run_burgers_six_seed.py --output {a.output} --plot-only\n'
            f'{sys.executable} runs/audit_burgers_six_seed.py --output {a.output}\n')
        archived=a.output/'code'; archived.mkdir()
        paths=list(source_config['source_hashes'])+['utils/burgers_six_seed.py','runs/run_burgers_six_seed.py',
            'runs/audit_burgers_six_seed.py','runs/BURGERS_SIX_SEED.md','tests/test_burgers_six_seed.py']
        for path in paths:
            dst=archived/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dst)
        reused=[]
        for sigma in SIGMAS:
            for seed in (0,1):
                src=a.source/f'sigma_{sigma:g}'/f'seed_{seed}'
                validate_condition(src,sigma,seed)
                dst=a.output/'raw'/f'sigma_{sigma:g}'/f'seed_{seed}'
                shutil.copytree(src,dst)
                files={str(x.relative_to(src)):sha(x) for x in src.rglob('*') if x.is_file()}
                for name,digest in files.items():
                    if sha(dst/name)!=digest: raise ValueError('Copy integrity failure')
                reused.append({'sigma':sigma,'seed':seed,'source':str(src.resolve()),'files':files})
        write_json(a.output/'reused_artifacts.json',reused)
    config=json.loads((a.output/'config.json').read_text())
    if config['seeds']!=list(SEEDS) or config['steps']!=STEPS: raise ValueError('Incompatible output configuration')
    for path,digest in config['source_hashes'].items():
        if sha(ROOT/path)!=digest: raise ValueError(f'Training source changed: {path}')
    jobs=[]; accounted=[]
    for sigma in SIGMAS:
        for seed in SEEDS:
            folder=a.output/'raw'/f'sigma_{sigma:g}'/f'seed_{seed}'
            if (folder/'summary.json').exists():
                validate_condition(folder,sigma,seed,observations=False)
                status='complete'
            elif folder.exists():
                raise ValueError(f'Incomplete run preserved; explicit recovery required: {folder}')
            else:
                jobs.append((folder,sigma,seed,700000+seed,STEPS));status='pending'
            accounted.append({'sigma':sigma,'seed':seed,'status':status})
    write_json(a.output/'run_status.json',accounted)
    if a.prepare_only:return
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(train_condition,*job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            folder,sigma,seed,_,_=futures[future]
            future.result();validate_condition(folder,sigma,seed)
            for r in accounted:
                if r['sigma']==sigma and r['seed']==seed:r['status']='complete'
            write_json(a.output/'run_status.json',accounted)
            from utils.burgers_six_seed import publish
            publish(a.output,require_all=False)
    from utils.burgers_six_seed import publish
    publish(a.output,require_all=True)

if __name__=='__main__':main()
