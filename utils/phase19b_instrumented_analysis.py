"""Fixed-grid, post-replay measurements. Never modifies replay states."""
import json
from pathlib import Path
import numpy as np
import torch
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from utils.burgers_recoverability import surrogate, library, PRIMITIVES, NAMES, P22, field_metrics
from utils.derivative_utils import build_burgers_reference_derivative_grids, compute_error_metrics
from utils.diagnostic_io import write_json, verify_inventory, sha256, inventory
from utils.transition_geometry import fit, geometry, basis, cosine
from utils.transition_update_geometry import gradients, update_metrics, arr
from utils.transition_instrumented_io import read_csv,write_csv
from utils.phase19b_instrumented_replay import SOURCE,PRIOR,ROOT


def analyze(out):
    out=Path(out);validation=json.loads((out/'validation.json').read_text())
    if validation['status']!='passed':raise RuntimeError('Unvalidated replay: interpretation prohibited')
    if (out/'transition_metrics.csv').exists():raise FileExistsError('Analysis already exists; preserve it')
    torch.set_num_threads(2)
    config=json.loads((out/'config.json').read_text());prov=json.loads((out/'provenance.json').read_text())
    matched=torch.load(P22,map_location='cpu',weights_only=False);scales=matched['scales']
    t,x,u=[matched[k] for k in ['t','x','u']];tf=t.numpy().ravel();xf=x.numpy().ravel()
    ts=np.unique(tf);xs=np.unique(xf);shape=(len(ts),len(xs));assert shape==(252,256)
    assert np.array_equal(tf,np.repeat(ts,len(xs))) and np.array_equal(xf,np.tile(xs,len(ts)))
    refs={k:v['physical'] for k,v in build_burgers_reference_derivative_grids(u_grid=u.numpy().reshape(shape),x_grid=xs,nu=.02).items() if k in PRIMITIVES+['u_t']}
    ar=library(np.column_stack([refs[k].ravel() for k in PRIMITIVES]));yr=refs['u_t'].ravel()
    ti=[NAMES.index('u*u_x'),NAMES.index('u_xx')];sp=[i for i in range(9) if i not in ti]
    truth=np.zeros(9);truth[ti]=[-1,.02]
    model=surrogate();sym=MinimalSymNet();assert list(product_term_dict(sym))==NAMES
    fixed=torch.tensor(np.floor(np.arange(4096)*len(tf)/4096),dtype=torch.long)
    np.save(out/'fixed_gradient_indices.npy',fixed.numpy())
    hist={int(r['step']):r for r in read_csv(out/'optimization_history.csv')}
    rows=[];lsrows=[];grows=[];geomrows=[];geoms={'reference':geometry(ar)}
    fieldsout={'x':xs,'t':ts,**{'reference__'+k:v for k,v in refs.items()}}
    diagnostic=[5000]+config['diagnostic_steps'];prevxi=None;prevstep=None;checks=[]
    prior=json.loads((PRIOR/'metrics.json').read_text())
    for step in diagnostic:
        state=torch.load(out/'states'/f'state_{step:06d}.pt',map_location='cpu',weights_only=False)
        update=torch.load(out/'updates'/f'update_{step:06d}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['surrogate']);sym.load_state_dict(state['symnet'])
        fields,canonical=field_metrics(model,tf,xf,refs)
        fields={k:v.astype(np.float64) for k,v in fields.items()}
        a=library(np.column_stack([fields[k].ravel() for k in PRIMITIVES]));yp=fields['u_t'].ravel()
        row={k:(v=='True' if 'recovery' in k else float(v)) for k,v in hist[step].items()};row['step']=step
        quantities={**{k:(fields[k],refs[k]) for k in PRIMITIVES+['u_t']},
                    'transport':(a[:,ti[0]],ar[:,ti[0]]),'weighted_diffusion':(.02*a[:,ti[1]],.02*ar[:,ti[1]]),
                    'rhs':(a@truth,yr)}
        for key,(pred,ref) in quantities.items():
            metrics=canonical[key] if key in PRIMITIVES else compute_error_metrics(pred,ref)
            for metric in ['mse','rmse','rel_l2']:row[key+'_'+metric]=float(metrics[metric])
            row[key+'_cosine']=cosine(pred,ref);row[key+'_pearson']=cosine(pred-pred.mean(),ref-ref.mean())
            row[key+'_prediction_rms']=float(np.sqrt(np.mean(pred**2)));row[key+'_reference_rms']=float(np.sqrt(np.mean(ref**2)))
        ec=a[:,ti[0]]-ar[:,ti[0]];ed=a[:,ti[1]]-ar[:,ti[1]];et=yp-yr;er=-ec+.02*ed;cl=et-er
        row.update(transport_error_norm=float(np.linalg.norm(ec)),diffusion_error_norm=float(np.linalg.norm(ed)),
                   weighted_diffusion_error_norm=float(np.linalg.norm(.02*ed)),error_transport_diffusion_cosine=cosine(ec,ed),
                   rhs_error_norm=float(np.linalg.norm(er)),rhs_cancellation_ratio=float(np.sum(er**2)/(np.sum(ec**2)+.02**2*np.sum(ed**2))),
                   closure_mse=float(np.mean(cl**2)),closure_norm=float(np.linalg.norm(cl)),closure_relative_to_predicted_ut=float(np.linalg.norm(cl)/np.linalg.norm(yp)))
        xi=np.array([row['xi_'+n] for n in NAMES])
        row.update(fixed_grid_data_loss=row['u_mse'],fixed_grid_pde_loss=float(np.mean((yp-a@xi)**2)))
        row['fixed_grid_total_loss']=row['fixed_grid_data_loss']+.5*row['fixed_grid_pde_loss']
        row['coefficient_interval_norm']=float(np.linalg.norm(xi-prevxi)) if prevxi is not None else 0.
        row['coefficient_interval_steps']=step-prevstep if prevstep is not None else 0
        prevxi=xi;prevstep=step
        for label,mat,target,gt,names in [('learned_learned',a,yp,truth,NAMES),('learned_reference',a,yr,truth,NAMES),
                                        ('reference_learned',ar,yp,truth,NAMES),('true_support_learned',a[:,ti],yp,truth[ti],[NAMES[i] for i in ti]),
                                        ('true_support_reference',a[:,ti],yr,truth[ti],[NAMES[i] for i in ti])]:
            rec,_=fit(mat,target,gt);coeff=np.zeros(9)
            for n,c in zip(names,rec.pop('coefficients')):coeff[NAMES.index(n)]=c
            lsrows.append({'step':step,'fit':label,**rec,**{'xi_'+n:float(c) for n,c in zip(NAMES,coeff)},
                           'spurious_l2':float(np.linalg.norm(coeff[sp])),
                           'loose_recovery':bool(abs(coeff[4]+1)<.25 and abs(coeff[2]-.02)<.02 and np.linalg.norm(coeff[sp])<.25),
                           'strong_recovery':bool(abs(coeff[4]+1)<.1 and abs(coeff[2]-.02)<.01 and np.linalg.norm(coeff[sp])<.1)})
        if step in config['geometry_steps']:
            g=geometry(a);q1=basis(a[:,ti]);q2=basis(a[:,sp])
            ang=np.degrees(np.arccos(np.clip(np.linalg.svd(q1.T@q2,compute_uv=False),0,1)))
            g['principal_angles_degrees']=ang.tolist();geoms[str(step)]=g
            geomrows.append({'step':step,'raw_condition':g['raw']['condition'],'normalized_condition':g['normalized']['condition'],
                             'raw_rank_1e12':g['raw']['ranks']['1e-12'],'normalized_rank_1e12':g['normalized']['ranks']['1e-12'],
                             'principal_angle_min':float(min(ang)),'principal_angle_max':float(max(ang)),
                             **{f'cos_{NAMES[i]}__{NAMES[j]}':g['normalized_gram'][i][j] for i in ti for j in sp}})
        for sample,indices in [('training',state['batch_indices']),('fixed',fixed)]:
            losses,gr=gradients(model,sym,scales,t,x,u,indices)
            g=update_metrics(model,sym,update,gr)
            grows.append({'step':step,'sample':sample,**losses,**g})
            if sample=='training':
                for k,v in losses.items():assert np.isclose(v,row[k],rtol=1e-6,atol=1e-8)
                errs={}
                for name in ['surrogate','symnet']:
                    gt=gr[name][0]+.5*gr[name][1];stored=arr(update[name+'_total_gradient'])
                    err=float(np.linalg.norm(gt-stored)/np.linalg.norm(stored));assert err<1e-4,(step,name,err)
                    errs[name+'_gradient_sum_relative_error']=err
                checks.append({'step':step,**errs})
            assert all(p.grad is None for p in list(model.parameters())+list(sym.parameters()))
        if step in [5000,10000]:
            for q in PRIMITIVES+['u_t','transport','rhs']:
                oldq='u*u_x' if q=='transport' else q
                old=next(r for r in prior['fidelity'] if r['state']==str(step) and r['quantity']==oldq)
                assert np.isclose(row[q+'_mse'],old['mse'],rtol=1e-5,atol=1e-9)
            assert np.isclose(row['rhs_cancellation_ratio'],prior['errors'][str(step)]['rhs_cancellation_ratio'],rtol=1e-6)
        rows.append(row)
        if step in [5000,7000,7500,8000,8500,9000,10000,10500]:
            fieldsout.update({str(step)+'__'+k:v for k,v in fields.items()})
        if step%250==0 or step in [8131,8631,8853,10013]:print('analyzed',step,flush=True)
    for name,data in [('transition_metrics',rows),('gradient_update_metrics',grows),('ls_trajectory',lsrows),('geometry_metrics',geomrows)]:write_csv(out/(name+'.csv'),data)
    write_json(out/'geometry.json',geoms);np.savez_compressed(out/'snapshot_fields.npz',**fieldsout)
    verify_inventory(SOURCE,prov['originals']['long_horizon']);verify_inventory(PRIOR,prov['originals']['posthoc'])
    assert sha256(ROOT/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb')==prov['notebook_sha256']
    validation.update(analysis_passed=True,offline_training_batch_losses_reproduced=True,gradient_sum_checks=checks,
                      canonical_endpoint_and_cancellation_checks_passed=True,diagnostic_count=len(rows),
                      source_artifacts_unchanged_after_analysis=True,analysis_source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'utils/transition_update_geometry.py']})
    write_json(out/'validation.json',validation)
    print('ANALYSIS COMPLETE',len(rows),'states',flush=True)
