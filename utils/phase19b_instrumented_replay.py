"""Exact replay with passive state/update capture; diagnostics run offline."""
import copy
import csv
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
from prog.minimal_symnet import MinimalSymNet
from utils.burgers_recoverability import surrogate, P22
from utils.phase19b_long_horizon import _coefficients, _metrics, NAMES
from utils.physics import phase19b_batch_losses
from utils.diagnostic_io import inventory, verify_inventory, write_json, sha256, state_hash
from utils.transition_instrumented_io import read_csv, write_csv

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'run_results/phase19b_long_horizon_control'
PRIOR=ROOT/'run_results/phase19b_transition_diagnostic'
OUT=ROOT/'run_results/phase19b_transition_instrumented'
DIAGNOSTIC_STEPS=sorted(set(range(7000,10501,25))|{8131,8631,8853,10013})
SAVE_STEPS=sorted(set(DIAGNOSTIC_STEPS)|{0,50,200,500,1000,2000,5000})
GEOMETRY_STEPS=[5000,7000,7500,8000,8250,8500,8750,9000,9500,10000,10500]
END_STEP=10501


def flat(params):return torch.cat([p.detach().reshape(-1) for p in params])


def replay(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir();(out/'updates').mkdir()
    torch.set_num_threads(2)
    originals={'long_horizon':inventory(SOURCE),'posthoc':inventory(PRIOR)}
    notebook=ROOT/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb'
    provenance={'originals':originals,'notebook_sha256':sha256(notebook),'matched_inputs_sha256':sha256(P22),
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'utils/physics.py',ROOT/'utils/phase19b_long_horizon.py']}}
    write_json(out/'provenance.json',provenance)
    config=json.loads((SOURCE/'config.json').read_text())
    config.update(phase='Phase 19B transition instrumentation',replay_end_step=END_STEP,diagnostic_steps=DIAGNOSTIC_STEPS,
                  saved_steps=SAVE_STEPS,geometry_steps=GEOMETRY_STEPS,evaluation_grid='entire archived 252x256 grid, unweighted',
                  gradient_samples=['actual training batch','fixed 4096-point systematic sample'],
                  fixed_gradient_indices='floor(arange(4096)*64512/4096)',capture='pre-update state; actual update step s to s+1',
                  diagnostics='offline after full replay validation; no diagnostic autograd during training',
                  validation_atol=1e-8,validation_rtol=1e-6,threads=2,
                  event_protocol={'fractional_progress':[.1,.25,.5,.75,.9],
                    'baseline':7000,'endpoint':10500,'sustained_steps':100,
                    'meaning':'fractional-progress milestones, NOT unique physical onsets; use regular 25-step grid',
                    'rapid_transport':'first forward 100-step slope of -xi_transport >= 5e-4 per step sustained for 100 start steps',
                    'gradient_sign':'first nonnegative data/PDE cosine sustained 100 steps; fixed sample primary',
                    'gradient_ratio':'first weighted PDE/data norm ratio <=1 sustained 100 steps',
                    'gradient_update_sign':'first update cosine with -data and -PDE both nonnegative sustained 100 steps',
                    'geometry_screen':'adjacent scheduled normalized condition relative change >20% or minimum angle absolute change >5 degrees; descriptive screen, not significance test'})
    write_json(out/'config.json',config)
    matched=torch.load(P22,map_location='cpu',weights_only=False)
    scales=matched['scales'].detach().clone();assert scales.tolist()==config['scales']
    t,x,u=[matched[k] for k in ['t','x','u']]
    generator=torch.Generator(device='cpu');generator.manual_seed(19220)
    batches=matched['batches']
    for old in batches:
        assert torch.equal(old,torch.randint(0,len(t),(4096,),generator=generator))
    model=surrogate();sym=MinimalSymNet()
    model.load_state_dict(matched['surrogate']);sym.load_state_dict(matched['symnet'])
    optimizer=torch.optim.Adam(list(model.parameters())+list(sym.parameters()),lr=5e-4)
    archived=read_csv(SOURCE/'optimization_history.csv')
    keys=['data_loss','pde_loss','total_loss']+['xi_'+n for n in NAMES]+['coefficient_error','spurious_l2']
    maxerr={k:0. for k in keys};maxstep={k:0 for k in keys};phaseerr={k:0. for k in keys}
    state_checks=[];flags={'first_loose_step':None,'first_strong_step':None}
    val={'status':'running','atol':1e-8,'rtol':1e-6,'compared_every_step':True}
    with (out/'optimization_history.csv').open('w',newline='') as stream, (out/'reproduction_errors.csv').open('w',newline='') as errorstream:
        writer=None;ew=csv.DictWriter(errorstream,fieldnames=['step']+keys,lineterminator='\n');ew.writeheader()
        for step in range(END_STEP+1):
            if step<len(batches):idx=batches[step-1] if step else batches[0]
            elif step==len(batches):idx=batches[-1]
            else:idx=torch.randint(0,len(t),(4096,),generator=generator)
            data,pde,total=phase19b_batch_losses(model,sym,scales,t,x,u,idx)
            xi=_coefficients(sym,scales.numpy());ce,sp,loose,strong=_metrics(xi)
            row={'step':step,'data_loss':float(data.detach()),'pde_loss':float(pde.detach()),'total_loss':float(total.detach()),
                 **dict(zip(['xi_'+n for n in NAMES],map(float,xi))), 'coefficient_error':ce,'spurious_l2':sp,'loose_recovery':loose,'strong_recovery':strong}
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row),lineterminator='\n');writer.writeheader()
            writer.writerow(row)
            errors={k:abs(row[k]-float(archived[step][k])) for k in keys};ew.writerow({'step':step,**errors})
            for k,e in errors.items():
                if e>maxerr[k]:maxerr[k]=e;maxstep[k]=step
                if 5000<=step<=10000:phaseerr[k]=max(phaseerr[k],e)
            bad=[k for k in keys if errors[k]>1e-8+1e-6*abs(float(archived[step][k]))]
            bad += [k for k in ['loose_recovery','strong_recovery'] if row[k]!=(archived[step][k]=='True')]
            if bad:
                val.update(status='failed',failed_step=step,failed_keys=bad,errors=errors)
                write_json(out/'validation.json',val)
                verify_inventory(SOURCE,originals['long_horizon']);verify_inventory(PRIOR,originals['posthoc'])
                raise RuntimeError(f'Reproduction failed at {step}: {bad}; mechanistic analysis prohibited')
            for label,value in [('loose',loose),('strong',strong)]:
                if value and flags['first_'+label+'_step'] is None:flags['first_'+label+'_step']=step
            if (SOURCE/f'checkpoint_{step:06d}.pt').exists():
                old=torch.load(SOURCE/f'checkpoint_{step:06d}.pt',map_location='cpu',weights_only=False)
                state_checks.append({'step':step,'surrogate_bitwise_equal':state_hash(model.state_dict())==state_hash(old['surrogate']),
                                     'symnet_bitwise_equal':state_hash(sym.state_dict())==state_hash(old['symnet'])})
            if step in SAVE_STEPS:
                rng=generator.get_state().clone();global_rng=torch.get_rng_state().clone()
                torch.save({'step':step,'surrogate':model.state_dict(),'symnet':sym.state_dict(),'optimizer':optimizer.state_dict(),
                            'batch_indices':idx,'generator_state_after_batch':rng,'global_rng':global_rng},out/'states'/f'state_{step:06d}.pt')
                before_m=flat(model.parameters()).clone();before_s=flat(sym.parameters()).clone()
                assert torch.equal(rng,generator.get_state()) and torch.equal(global_rng,torch.get_rng_state())
            if step==END_STEP:break
            optimizer.zero_grad(set_to_none=True);total.backward();optimizer.step()
            if step in SAVE_STEPS:
                torch.save({'surrogate_delta':flat(model.parameters())-before_m,'symnet_delta':flat(sym.parameters())-before_s,
                            'surrogate_total_gradient':torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]),
                            'symnet_total_gradient':torch.cat([p.grad.detach().reshape(-1) for p in sym.parameters()]),
                            'optimizer_after':optimizer.state_dict()},out/'updates'/f'update_{step:06d}.pt')
                assert torch.equal(rng,generator.get_state()) and torch.equal(global_rng,torch.get_rng_state())
            if step%500==0:
                stream.flush();errorstream.flush();print(f'replay {step}: transport={xi[4]:.8f}; max loss discrepancy={max(maxerr[k] for k in keys[:3]):.3g}',flush=True)
    verify_inventory(SOURCE,originals['long_horizon']);verify_inventory(PRIOR,originals['posthoc'])
    assert sha256(notebook)==provenance['notebook_sha256']
    val.update(status='passed',rows_compared=END_STEP+1,max_absolute_error=maxerr,max_error_step=maxstep,
               interval_5000_10000_max_absolute_error=phaseerr,checkpoint_state_checks=state_checks,
               original_artifacts_unchanged=True,notebook_unchanged=True,passive_capture_rng_unchanged=True,**flags)
    write_json(out/'validation.json',val)
    print('REPRODUCTION PASSED',flags,flush=True)
