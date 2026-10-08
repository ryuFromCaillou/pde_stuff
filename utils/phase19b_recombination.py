"""Frozen 2x2 intervention. No surrogate optimizer or trajectory reconstruction."""
import copy
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from utils.burgers_recoverability import (ROOT, P22, PRIMITIVES, NAMES, TRUTH, SPURIOUS,
    surrogate, library, coefficient_scales, coefficients, recovered, coordinate_check)
from utils.derivative_utils import evaluate_diagnostic_fields, compute_error_metrics
from utils.diagnostic_io import sha256, state_hash, write_json, inventory, verify_inventory
from utils.transition_geometry import fit

SOURCE=ROOT/'run_results/phase19b_transition_instrumented'
PAIRS={'A':(8000,8000),'B':(8000,8750),'C':(8750,8000),'D':(8750,8750)}

def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()

def metrics(xi):
    return dict(coefficient_error=float(np.linalg.norm(xi-TRUTH)),spurious_l2=float(np.linalg.norm(xi[SPURIOUS])),
                loose=recovered(xi),strong=recovered(xi,True),**{'xi_'+n:float(v) for n,v in zip(NAMES,xi)})

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    protected=[SOURCE,ROOT/'run_results/phase19b_long_horizon_control',ROOT/'run_results/phase19b_transition_diagnostic']
    inventories={str(p):inventory(p) for p in protected}
    notebook=ROOT/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb';nbhash=sha256(notebook)
    manifest=json.loads((SOURCE/'artifact_manifest.json').read_text())
    verify_inventory(SOURCE,manifest['artifacts_sha256'])
    oldvalidation=json.loads((SOURCE/'validation.json').read_text())
    assert oldvalidation['status']=='passed' and oldvalidation['analysis_passed']
    matched=torch.load(P22,map_location='cpu',weights_only=False)
    scales=matched['scales'].numpy().ravel();sourceconfig=json.loads((SOURCE/'config.json').read_text())
    assert np.array_equal(scales,np.array(sourceconfig['scales']))
    config={'pre':8000,'post':8750,'pairs':PAIRS,'seeds':list(range(25)), 'steps':1000,
      'optimizer':'Adam','lr':.01,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
      'objective':'unweighted full-grid mean squared PDE residual; unscaled physical u_t',
      'dtype':'float32','device':'cpu','threads':2,'scales':scales.tolist(),
      'scaling':'original Phase19B transferred scale vector fixed for ALL conditions; no checkpoint RMS re-estimation',
      'coefficient_conversion':'physical xi = expanded scaled coefficients / [s0,s1,s2,s0^2,s0*s1,s0*s2,s1^2,s1*s2,s2^2]',
      'names':NAMES,'truth':TRUTH.tolist(),'spurious_indices':SPURIOUS,
      'criteria':sourceconfig['recovery_criteria'],'strict_inequalities':True,
      'primary_endpoint':'after 1000 head optimizer updates; first crossings also recorded at every step including 0',
      'reliable_definition':'descriptive >=20/25 terminal loose recoveries; report Wilson 95% intervals; not a population guarantee',
      'initialization':'fresh MinimalSymNet default torch Linear uniform initialization, torch.manual_seed(seed), cloned A/B/C/D',
      'grid':'all 64512 archived points, time-major spatial-minor, physical coordinates, shape 252x256',
      'checkpoint_semantics':'exact captured pre-update states; no surrogate updates',
      'budget_basis':'existing Phase23/24 frozen-head protocol; fixed before outcomes; scaling deliberately original Phase19B',
      'post_control_caveat':'8750 precedes archived first loose LS 8825 and joint SymNet 8853; D is not known-positive at these thresholds'}
    write_json(out/'config.json',config)
    provenance={'source_inventories':inventories,'notebook_sha256':nbhash,'matched_inputs_sha256':sha256(P22),
      'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
      'implementation':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'runs/run_phase19b_recombination.py',ROOT/'prog/minimal_symnet.py',ROOT/'utils/burgers_recoverability.py',ROOT/'utils/derivative_utils.py']},
      'validated_replay':{'rows':oldvalidation['rows_compared'],'max_absolute_error':oldvalidation['max_absolute_error']}}
    write_json(out/'provenance.json',provenance)
    t,x=[matched[k].numpy().ravel() for k in ['t','x']];shape=(252,256)
    assert len(t)==64512 and np.array_equal(t,np.repeat(np.unique(t),256)) and np.array_equal(x,np.tile(np.unique(x),252))
    np.savez_compressed(out/'evaluation_grid.npz',t=t,x=x,indices=np.arange(len(t)))
    cache=np.load(SOURCE/'snapshot_fields.npz')
    archived=pd.read_csv(SOURCE/'transition_metrics.csv').set_index('step')
    archived_ls=pd.read_csv(SOURCE/'ls_trajectory.csv')
    fields={};physical={};theta={};targets={};metadata={};checks=[];controls=[]
    for step in [8000,8750]:
        path=SOURCE/f'states/state_{step:06d}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==step
        model=surrogate();model.load_state_dict(state['surrogate']);model.eval();model.requires_grad_(False)
        before=state_hash(model.state_dict())
        f={k:v.astype(np.float64) for k,v in evaluate_diagnostic_fields(model,t,x,shape,PRIMITIVES).items()}
        assert state_hash(model.state_dict())==before
        for k,v in f.items():
            if str(step)+'__'+k in cache:
                assert np.array_equal(v,cache[str(step)+'__'+k])
                checks.append({'step':step,'quantity':k,'cached_bitwise_equal':True})
            m=compute_error_metrics(v,cache['reference__'+k])
            for metric in ['mse','rel_l2']:
                actual=float(m[metric]);expected=float(archived.loc[step,k+'_'+metric])
                assert np.isclose(actual,expected,atol=1e-10,rtol=1e-7),(step,k,metric,actual,expected)
                checks.append({'step':step,'quantity':k,'metric':metric,'actual':actual,'archived':expected})
        fields[step]=f;physical[step]=np.column_stack([f[k].ravel() for k in PRIMITIVES]);theta[step]=library(physical[step]);targets[step]=f['u_t'].ravel()
        np.savez_compressed(out/f'frozen_{step}.npz',**f,Theta=theta[step])
        metadata[str(step)]={'state_file':str(path),'state_sha256':sha256(path),'surrogate_tensor_sha256':before,
          'shape':shape,'Theta_shape':theta[step].shape,'quantities':{k:{'shape':v.shape,'dtype':str(v.dtype),'sha256':array_hash(v)} for k,v in f.items()},
          'Theta_sha256':array_hash(theta[step]),'cached_fields_available':step==8000,
          'verification':'PRE cached fields bitwise; both states archive-manifest hash verified and canonical full-grid metrics and LS compared'}
        head=MinimalSymNet();head.load_state_dict(state['symnet']);xi=coefficients(head,scales)
        controls.append({'step':step,**metrics(xi),'pde_residual':float(np.mean((theta[step]@xi-targets[step])**2))})
    write_json(out/'frozen_metadata.json',metadata);pd.DataFrame(controls).to_csv(out/'historical_controls.csv',index=False)
    ls=[];pairchecks={}
    for cond,(s,tg) in PAIRS.items():
        a,y=theta[s],targets[tg]
        pairchecks[cond]={'spatial_state':s,'target_state':tg,'Theta_sha256':array_hash(a),'target_sha256':array_hash(y),'scales':scales.tolist()}
        for support,inds in [('full',list(range(9))),('true_only',[4,2])]:
            r,_=fit(a[:,inds],y,TRUTH[inds]);xi=np.zeros(9);xi[inds]=r.pop('coefficients')
            ls.append({'condition':cond,'support':support,**r,**metrics(xi)})
            if cond in ['A','D']:
                old=archived_ls[(archived_ls.step==s)&(archived_ls['fit']==('learned_learned' if support=='full' else 'true_support_learned'))].iloc[0]
                assert np.allclose(xi,[old['xi_'+n] for n in NAMES],atol=1e-10,rtol=1e-7)
        print('LS',cond,ls[-2]['xi_u*u_x'],ls[-2]['spurious_l2'],flush=True)
    assert pairchecks['B']['Theta_sha256']==pairchecks['A']['Theta_sha256']
    assert pairchecks['C']['Theta_sha256']==pairchecks['D']['Theta_sha256']
    assert pairchecks['B']['target_sha256']==pairchecks['D']['target_sha256']
    assert pairchecks['C']['target_sha256']==pairchecks['A']['target_sha256']
    write_json(out/'pairing_validation.json',pairchecks);pd.DataFrame(ls).to_csv(out/'ls_results.csv',index=False)
    validation={'status':'frozen_inputs_validated','source_manifest_verified':True,'field_checks':checks,
      'LS_controls_match_archive':True,'feature_order':NAMES,'fixed_grid_verified':True,'pairing_exact':True}
    write_json(out/'validation.json',validation)
    assert list(product_term_dict(MinimalSymNet()))==NAMES
    features={s:torch.tensor(physical[s],dtype=torch.float32)/torch.tensor(scales) for s in physical}
    ys={s:torch.tensor(targets[s,None] if False else targets[s].reshape(-1,1),dtype=torch.float32) for s in targets}
    assert all(not z.requires_grad for z in list(features.values())+list(ys.values()))
    (out/'heads').mkdir();(out/'trajectories').mkdir();finals=[];conversion=[]
    for seed in range(25):
        torch.manual_seed(seed);initial=MinimalSymNet().state_dict();ih=state_hash(initial)
        torch.save(initial,out/'heads'/f'initial_{seed:02d}.pt')
        for cond,(s,tg) in PAIRS.items():
            head=MinimalSymNet();head.load_state_dict(copy.deepcopy(initial));assert state_hash(head.state_dict())==ih
            opt=torch.optim.Adam(head.parameters(),lr=.01)
            assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in head.parameters()}
            conversion.append(coordinate_check(head,physical[s],scales))
            hist=[];first={'loose':None,'strong':None}
            for step in range(1001):
                loss=torch.nn.functional.mse_loss(head(features[s]),ys[tg]);assert torch.isfinite(loss)
                xi=coefficients(head,scales);m=metrics(xi)
                for key in first:
                    if m[key] and first[key] is None:first[key]=step
                hist.append({'condition':cond,'seed':seed,'step':step,'pde_residual':float(loss.detach()),**m})
                if step==1000:break
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            conversion.append(coordinate_check(head,physical[s],scales))
            direct=float(np.mean((theta[s]@xi-targets[tg])**2))
            assert np.isclose(direct,hist[-1]['pde_residual'],rtol=2e-5,atol=1e-8)
            result={**hist[-1],'first_loose_step':first['loose'],'first_strong_step':first['strong'],'initial_sha256':ih,
                    'physical_polynomial_residual':direct}
            finals.append(result);pd.DataFrame(hist).to_csv(out/'trajectories'/f'{cond}_seed_{seed:02d}.csv',index=False)
            torch.save(head.state_dict(),out/'heads'/f'{cond}_seed_{seed:02d}.pt')
            pd.DataFrame(finals).to_csv(out/'per_seed_results.csv',index=False)
        print(f'paired seeds {seed+1}/25 complete',flush=True)
    final=pd.DataFrame(finals);summary=[]
    for cond,g in final.groupby('condition'):
        r={'condition':cond,'n':len(g)}
        for key in ['loose','strong']:
            count=int(g[key].sum());lo,hi=wilson(count,len(g))
            r.update({key+'_count':count,key+'_fraction':count/len(g),key+'_ci_low':lo,key+'_ci_high':hi,
              key+'_ever_count':int(g['first_'+key+'_step'].notna().sum()),
              key+'_median_first_step_terminal_recovered':None if not count else float(g.loc[g[key],'first_'+key+'_step'].median())})
        for k in ['coefficient_error','spurious_l2','pde_residual']+['xi_'+n for n in NAMES]:
            for stat,val in [('median',g[k].median()),('q25',g[k].quantile(.25)),('q75',g[k].quantile(.75)),('min',g[k].min()),('max',g[k].max())]:r[k+'_'+stat]=float(val)
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'recovery_summary.csv',index=False)
    paired=[]
    from scipy.stats import binomtest
    for a,b in [('A','B'),('A','C'),('B','C'),('B','D'),('C','D')]:
        for key in ['loose','strong']:
            ga=final[final.condition==a].set_index('seed')[key];gb=final[final.condition==b].set_index('seed')[key]
            n10=int((ga&~gb).sum());n01=int((~ga&gb).sum())
            paired.append({'a':a,'b':b,'criterion':key,'a_only':n10,'b_only':n01,'both':int((ga&gb).sum()),'neither':int((~ga&~gb).sum()),
              'mcnemar_exact_p_unadjusted':float(binomtest(n10,n10+n01).pvalue) if n10+n01 else 1.})
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    for p in protected:verify_inventory(p,inventories[str(p)])
    assert sha256(notebook)==nbhash
    validation.update(status='passed',all_100_heads_complete=True,paired_initializations_identical=True,
      only_head_optimized=True,source_artifacts_unchanged=True,notebook_unchanged=True,
      max_physical_conversion_absolute_error=max(conversion),terminal_physical_residual_checks=True)
    write_json(out/'validation.json',validation)
    from utils.phase19b_recombination_report import render
    render(out)
    write_json(out/'artifact_manifest.json',inventory(out))
