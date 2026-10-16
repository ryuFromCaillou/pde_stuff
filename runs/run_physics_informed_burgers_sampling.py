"""Run the nested clean-observation Burgers sampling robustness study."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, multiprocessing, platform, shutil, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from utils.physics_informed_burgers import NOTEBOOKS, hashes, write_json
from utils.physics_informed_burgers_sampling import RATIOS,COUNTS,SEEDS,SAMPLING_SEED,validate_sampling_protocol,train_condition,finalize_control,make_figures

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('run_results/physics_informed_burgers/sampling_robustness')); p.add_argument('--steps',type=int,default=100000); p.add_argument('--workers',type=int,choices=(1,2,4),default=4); p.add_argument('--validate-only',action='store_true'); p.add_argument('--clean-control',type=Path,default=Path('run_results/physics_informed_burgers/clean_baseline')); a=p.parse_args()
 if not 1<=a.steps<=100000: p.error('steps must be between 1 and 100000')
 a.output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
 source=hashes((*NOTEBOOKS,'utils/physics_informed_burgers.py','utils/physics_informed_burgers_sampling.py','runs/run_physics_informed_burgers_sampling.py','prog/mlps.py','prog/featlib.py','utils/derivative_utils.py','Datasets/data/processed/burg_gen/burg_gen.py'))
 config=dict(experiment='Burgers sampling robustness',ratios=list(RATIOS),counts=list(COUNTS),seeds=list(SEEDS),sampling_seed=SAMPLING_SEED,steps=a.steps,dataset=dict(seed=0,nu=.02,T=.1,N=256,dt=.002,observations=640),sampling=dict(method='nested prefix of one uniform random permutation',denominator=640,batch_size='min(640,count)',with_replacement=True),training=dict(lr=5e-4,lambda_data=1.,lambda_pde=1.,l1=0.,optimizer='Adam',physical_head=True,no_least_squares=True),evaluation=dict(reference='clean 51x256 solver grid',coefficients=[-1.,.02]),execution=dict(noise=False,crossed_experiment_executed=False),source_hashes=source,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
 write_json(a.output/'config.json',config); (a.output/'reproduction.txt').write_text(f'{sys.executable} runs/run_physics_informed_burgers_sampling.py --output {a.output} --steps {a.steps} --workers {a.workers}\n')
 val=validate_sampling_protocol(); val['notebooks_unchanged']=hashes(NOTEBOOKS)=={k:source[k] for k in NOTEBOOKS}; write_json(a.output/'validation.json',val)
 if a.validate_only: print('Sampling protocol validation passed'); return
 results=[]; jobs=[]
 for ratio,count in zip(RATIOS,COUNTS):
  for seed in SEEDS:
   d=a.output/f'ratio_{ratio:g}'/f'seed_{seed}'
   if ratio==1.0 and not (d/'summary.json').exists(): results.append(finalize_control(d,a.clean_control/f'seed_{seed}'))
   elif (d/'summary.json').exists(): results.append(json.loads((d/'summary.json').read_text()))
   else: jobs.append((ratio,count,seed))
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  fs={pool.submit(train_condition,a.output/f'ratio_{r:g}'/f'seed_{s}',r,c,s,a.steps): (r,s) for r,c,s in jobs}
  for f in concurrent.futures.as_completed(fs): results.append(f.result()); write_json(a.output/'completed_conditions.json',sorted(results,key=lambda x:(-x.get('ratio',0),x.get('init_seed',x.get('seed',0)))))
 make_figures(a.output); val.update(completed_conditions=len(results),expected_conditions=10,trained_conditions=len(jobs),control_reused=True,all_conditions_complete=len(results)==10); write_json(a.output/'validation.json',val); print('Sampling experiment complete',flush=True)
if __name__=='__main__': main()
