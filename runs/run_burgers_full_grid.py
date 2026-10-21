"""Gated full-grid six-seed Burgers sampling study."""
import argparse,concurrent.futures,json,multiprocessing,platform,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from utils.physics_informed_burgers import NOTEBOOKS,write_json
from utils.burgers_full_grid import *

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,default=Path('run_results/physics_informed_burgers/sampling_full_grid_six_seed'))
    p.add_argument('--mode',choices=['prepare','smoke','gate','remaining','publish'],required=True)
    p.add_argument('--workers',type=int,default=4,choices=[1,2,4])
    a=p.parse_args();r=a.output;torch.set_num_threads(1)
    if a.mode=='prepare':
        r.mkdir(parents=True,exist_ok=False)
        paths=['notebook/diagnostics/pauls_tweaks_diag_notebook.ipynb','notebook/diagnostics/burgers_minimal_discovery_story.ipynb']+['utils/physics_informed_burgers.py','utils/physics_informed_burgers_sampling.py','utils/physics_informed_burgers_noise.py','utils/burgers_six_seed.py','utils/burgers_full_grid.py','utils/burgers_full_grid_reporting.py','runs/run_burgers_full_grid.py','runs/BURGERS_FULL_GRID.md','prog/mlps.py','prog/featlib.py','utils/derivative_utils.py','Datasets/data/processed/burg_gen/burg_gen.py']
        source={name:sha(ROOT/name) for name in paths}
        config=dict(experiment='Full-grid six-seed sampling convergence',ratios=RATIOS,counts=COUNTS,seeds=SEEDS,steps=100000,
            sampling_seed=SAMPLING_SEED,population=13056,sampling='one shared nested permutation of time-major full grid',
            dataset=dict(seed=0,nu=.02,T=.1,dt=.002,N=256,shape=[51,256]),
            training=dict(optimizer='Adam',lr=.0005,data_weight=1,pde_weight=1,noise=0,batch_size=640,with_replacement=True,
                batch_seed='10000+initialization seed',architecture='SirenMLP hidden_size=64 hidden_layers=3 omega=1/1',
                physical_head='zero initialized unconstrained two coefficients',l1=0,symnet=False,least_squares=False),
            recovery=dict(targets=[-1,.02],loose_abs=[.25,.02],strong_abs=[.1,.01],strict=True,first_crossing_not_sustained=True),
            checkpoints=CHECKPOINTS,source_hashes=source,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu',dtype='float32',threads_per_worker=1),
            historical_reference='640 structured observations = 4.9019608% of full grid; not a random full-grid subset',
            interpretation='Observation density and PDE collocation coverage vary together. Fixed solver and evaluation grid; known-support parameter estimation.')
        write_json(r/'config.json',config)
        for name in source:
            dst=r/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        protected={}
        for path in (ROOT/'run_results/physics_informed_burgers').iterdir():
            if path!=r.resolve() and path.name!=r.name:
                for f in path.rglob('*'):
                    if f.is_file():protected[str(f.relative_to(ROOT))]=sha(f)
        write_json(r/'protected_artifacts.json',protected)
        write_json(r/'validation.json',validate_protocol())
        (r/'reproduction.txt').write_text('\n'.join(f'{sys.executable} runs/run_burgers_full_grid.py --output {r} --mode {mode}' for mode in ['prepare','smoke','gate','remaining','publish'])+'\n')
        return
    config=json.loads((r/'config.json').read_text())
    for name,digest in config['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    if a.mode=='smoke':
        for ratio,count in [(1.,13056),(.1,1306)]:
            folder=r/'smoke'/f'ratio_{ratio:g}'/'seed_0'
            train_condition(folder,ratio,count,0,100)
            validate_condition(folder,ratio,count,0,100)
        write_json(r/'smoke_validation.json',dict(passed=True,conditions=2,steps=100))
    elif a.mode=='gate':
        assert json.loads((r/'smoke_validation.json').read_text())['passed']
        folder=r/'raw'/'ratio_1'/'seed_0'
        train_condition(folder,1.,13056,0,100000)
        h,m=validate_condition(folder,1.,13056,0)
        write_json(r/'gate_validation.json',dict(passed=True,summary=json.loads((folder/'summary.json').read_text()),finite=True,checkpoint_losses_reproduced=True))
    elif a.mode=='remaining':
        gate=json.loads((r/'gate_review.json').read_text());assert gate['approved_for_remaining']
        jobs=[];status=[]
        for ratio,count in zip(RATIOS,COUNTS):
            for seed in SEEDS:
                folder=r/'raw'/f'ratio_{ratio:g}'/f'seed_{seed}'
                if (folder/'summary.json').exists():validate_condition(folder,ratio,count,seed);state='validated'
                elif folder.exists():raise ValueError(f'Partial run preserved: {folder}')
                else:jobs.append((folder,ratio,count,seed,100000));state='pending'
                status.append(dict(ratio=ratio,seed=seed,status=state))
        write_json(r/'run_status.json',status)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(train_condition,*job):job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                folder,ratio,count,seed,_=futures[future];future.result();validate_condition(folder,ratio,count,seed)
                for entry in status:
                    if entry['ratio']==ratio and entry['seed']==seed:entry['status']='validated'
                write_json(r/'run_status.json',status)
        from utils.burgers_full_grid_reporting import publish
        publish(r)
    elif a.mode=='publish':
        from utils.burgers_full_grid_reporting import publish
        publish(r)
if __name__=='__main__':main()
