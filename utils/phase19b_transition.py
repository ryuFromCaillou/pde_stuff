"""Post-hoc Phase 19B checkpoint evaluation; no optimizer or training calls."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from utils.burgers_recoverability import surrogate, library, coefficients, PRIMITIVES, NAMES, P22, field_metrics
from utils.derivative_utils import build_burgers_reference_derivative_grids, compute_error_metrics
from utils.diagnostic_io import inventory, verify_inventory, write_json, sha256
from utils.transition_geometry import geometry, fit, basis, project, cosine

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'run_results/phase19b_long_horizon_control'
DEFAULT_OUT = ROOT/'run_results/phase19b_transition_diagnostic'

def run(out=DEFAULT_OUT):
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    before=inventory(SOURCE)
    notebook=ROOT/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb'
    notebook_hash=sha256(notebook)
    matched=torch.load(P22,map_location='cpu',weights_only=False)
    tf=matched['t'].numpy().ravel(); xf=matched['x'].numpy().ravel()
    t=np.unique(tf); x=np.unique(xf); shape=(len(t),len(x))
    assert np.array_equal(tf,np.repeat(t,len(x))) and np.array_equal(xf,np.tile(x,len(t)))
    u=matched['u'].numpy().reshape(shape)
    refs={k:v['physical'] for k,v in build_burgers_reference_derivative_grids(u_grid=u,x_grid=x,nu=.02).items() if k in PRIMITIVES+['u_t']}
    names=list(product_term_dict(MinimalSymNet()))
    assert names==NAMES
    true_idx=[names.index('u*u_x'),names.index('u_xx')]
    sp=[i for i in range(len(names)) if i not in true_idx]
    truth=np.zeros(len(names));truth[true_idx]=[-1,.02]
    fields={'reference':refs};heads={};canonical={}
    with (SOURCE/'checkpoint_metrics.csv').open() as f: archived={r['step']:r for r in csv.DictReader(f)}
    checks=[]
    for step in [5000,10000,20000]:
        state=torch.load(SOURCE/f'checkpoint_{step:06d}.pt',map_location='cpu',weights_only=False)
        assert state['step']==step
        model=surrogate();model.load_state_dict(state['surrogate']);model.eval()
        f,m=field_metrics(model,tf,xf,refs)
        fields[str(step)]={k:np.asarray(v,dtype=np.float64) for k,v in f.items()};canonical[str(step)]=m
        sym=MinimalSymNet();sym.load_state_dict(state['symnet'])
        heads[str(step)]=coefficients(sym,matched['scales'].numpy()).tolist()
        for k in PRIMITIVES+['u_t']:
            actual=float(np.mean((f[k]-refs[k])**2));expected=float(archived[str(step)][k+'_mse'])
            assert np.isclose(actual,expected,rtol=2e-5,atol=1e-9),(step,k,actual,expected)
            checks.append({'step':step,'field':k,'actual':actual,'archived':expected})
        print('evaluated checkpoint',step,flush=True)
    matrices={k:library(np.column_stack([f[n].ravel() for n in PRIMITIVES])) for k,f in fields.items()}
    yr=refs['u_t'].ravel(); ar=matrices['reference'];metrics=[];geom={};fits=[];errors={};projections=[];alignments=[]
    for label,f in fields.items():
        a=matrices[label];geom[label]=geometry(a)
        qt=basis(a[:,true_idx]);qs=basis(a[:,sp]);qr=basis(ar[:,true_idx]);qrs=basis(ar[:,sp])
        angles=np.degrees(np.arccos(np.clip(np.linalg.svd(qt.T@qs,compute_uv=False),0,1)))
        geom[label]['true_spurious_principal_angles_degrees']=angles.tolist()
        for j in true_idx:
            for i in sp:
                alignments.append({'state':label,'true_term':names[j],'spurious_term':names[i],
                                   'cosine':cosine(a[:,j],a[:,i]),'pearson':geom[label]['pearson'][j][i]})
        if label=='reference': targets={'reference':yr}
        else: targets={'predicted':f['u_t'].ravel(),'reference':yr}
        for target,y in targets.items():
            full,res=fit(a,y,truth); two,r2=fit(a[:,true_idx],y,truth[true_idx])
            residual_sp=a[:,sp]-qt@(qt.T@a[:,sp]);qi=basis(residual_sp)
            for support,record in [('full',full),('true_only',two)]:
                record.update(state=label,target=target,support=support)
                fits.append(record)
            full['true_minus_full_mse']=two['residual_mse']-full['residual_mse']
            full['incremental_spurious_fraction_of_true_residual']=project(r2,qi)['energy_fraction'] if two['residual_relative_l2'] > 1e-12 else None
            assert full['residual_mse']<=two['residual_mse']+1e-12
            if label!='reference':
                # Reference features + predicted target separates target error from feature error observationally.
                if target=='predicted':
                    rec,_=fit(ar,y,truth);rec.update(state=label,target='predicted_on_reference_features',support='full');fits.append(rec)
        if label=='reference':continue
        for quantity in PRIMITIVES+['u_t','u*u_x','rhs']:
            pred=(a@truth).reshape(shape) if quantity=='rhs' else (a[:,names.index(quantity)].reshape(shape) if quantity=='u*u_x' else f[quantity])
            ref=yr.reshape(shape) if quantity=='rhs' else (ar[:,names.index(quantity)].reshape(shape) if quantity=='u*u_x' else refs[quantity])
            metric=compute_error_metrics(pred,ref)
            metric.update(state=label,quantity=quantity,cosine=cosine(pred,ref),pearson=cosine(pred-pred.mean(),ref-ref.mean()),
                          rms_prediction=float(np.sqrt(np.mean(pred**2))),rms_reference=float(np.sqrt(np.mean(ref**2))))
            metrics.append(metric)
        et=f['u_t'].ravel()-yr;ec=a[:,true_idx[0]]-ar[:,true_idx[0]];ed=a[:,true_idx[1]]-ar[:,true_idx[1]]
        erhs=-ec+.02*ed;closure=et-erhs
        vectors={'e_t':et,'e_transport':ec,'e_diffusion':ed,'e_rhs':erhs,'closure_error':closure}
        errors[label]={'pairwise_cosine':{f'{n},{m}':cosine(v,w) for n,v in vectors.items() for m,w in vectors.items()},
                       'rhs_error_mse':float(np.mean(erhs**2)), 'closure_mse':float(np.mean(closure**2)),
                       'uncancelled_rhs_error_energy':float(np.mean(ec**2)+.02**2*np.mean(ed**2)),
                       'rhs_cancellation_ratio':float(np.mean(erhs**2)/(np.mean(ec**2)+.02**2*np.mean(ed**2))),
                       'target_rhs_error_cancellation_ratio':float(np.mean(closure**2)/(np.mean(et**2)+np.mean(erhs**2))),
                       'trained_head_mse':float(np.mean((a@heads[label]-f['u_t'].ravel())**2)),
                       'et_column_projection':{names[i]:{'cosine':cosine(et,a[:,i]),'energy_fraction':cosine(et,a[:,i])**2,
                                                       'physical_projection_coefficient':float(a[:,i]@et/(a[:,i]@a[:,i]))} for i in range(len(names))}}
        for n,v in vectors.items():
            for space,q in [('predicted_true',qt),('predicted_spurious',qs),('reference_true',qr),('reference_spurious',qrs)]:
                projections.append(dict(state=label,error=n,space=space,**project(v,q)))
        # Error allocation in the joint library: y - A*truth = e_t + e_transport - .02*e_diffusion.
        bias={}
        for n,v in [('target',et),('transport',ec),('diffusion',-.02*ed)]:
            z,_=fit(a,v,np.zeros(len(names)));bias[n]=z['coefficients']
        errors[label]['coefficient_bias_components']=bias
        full=next(z for z in fits if z['state']==label and z['target']=='predicted' and z['support']=='full')
        assert np.allclose(np.sum(list(bias.values()),axis=0),np.array(full['coefficients'])-truth,atol=1e-9)
    result={'names':names,'true_indices':true_idx,'spurious_indices':sp,'truth':truth.tolist(),'grid_shape':shape,
            'geometry':geom,'fidelity':metrics,'fits':fits,'errors':errors,'projections':projections,'alignments':alignments,
            'trained_heads':heads,'canonical_primitive_metrics':canonical}
    write_json(out/'metrics.json',result)
    np.savez_compressed(out/'fields.npz',x=x,t=t,**{label+'__'+k:v for label,f in fields.items() for k,v in f.items()})
    for table,data in [('fidelity',metrics),('projections',projections),('alignments',alignments)]:
        with (out/(table+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_json(out/'config.json',{'source':str(SOURCE),'checkpoints':[5000,10000,20000],'grid_shape':shape,'grid_points':len(tf),
                                'reference':'archived observations; centered periodic generator differences; u_t is Burgers RHS',
                                'linear_algebra_dtype':'float64','svd_relative_cutoff':1e-12,'normalization':'unit column L2 norm, no centering',
                                'training_performed':False,'matched_inputs_sha256':sha256(P22)})
    verify_inventory(SOURCE,before);assert sha256(notebook)==notebook_hash
    write_json(out/'validation.json',{'source_inventory':before,'source_unchanged':True,'notebook_unchanged':True,'checkpoint_mse_checks':checks,
                                    'feature_order_verified':True,'coefficient_bias_identity_verified':True})
    print('measurements saved',out,flush=True)
