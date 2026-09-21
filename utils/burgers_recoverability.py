"""Matched smooth-Burgers experiment machinery, shared with artifact analysis.

The scientific design and public output contract are in runs/PHASE24.md.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.mlps import SirenMLP
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from prog.featlib import FeatureTensor
from utils.derivative_utils import (evaluate_diagnostic_fields, evaluate_primitive_feature_metrics,
    compute_error_metrics, periodic_regularity_metrics, build_burgers_reference_derivative_grids)
from utils.diagnostic_io import sha256, state_hash, write_json, inventory, verify_inventory

CHECKPOINTS = [0, 50, 200, 500, 1000]
PRIMITIVES = ['u', 'u_x', 'u_xx']
NAMES = ['u', 'u_x', 'u_xx', 'u^2', 'u*u_x', 'u*u_xx', 'u_x^2', 'u_x*u_xx', 'u_xx^2']
TRUTH = np.array([0., 0., .02, 0., -1., 0., 0., 0., 0.])
SPURIOUS = [i for i, v in enumerate(TRUTH) if v == 0]
ROOT = Path(__file__).resolve().parents[1]
P23 = ROOT / 'run_results/phase23_joint_trajectory_recoverability'
P22 = ROOT / 'run_results/phase22_sgd_control/matched_inputs.pt'
OUT = ROOT / 'run_results/phase24_smooth_burgers_control'


def surrogate():
    return SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=20., hidden_omega_0=1.)


def rms(fields):
    return np.maximum(np.sqrt(np.mean(np.column_stack([fields[n].ravel() for n in PRIMITIVES]) ** 2, axis=0)), 1e-12)


def coefficient_scales(s):
    return np.array([s[0], s[1], s[2], s[0]**2, s[0]*s[1], s[0]*s[2], s[1]**2, s[1]*s[2], s[2]**2], dtype=np.float64)


def coefficients(model, scales):
    return np.array(list(product_term_dict(model).values())) / coefficient_scales(scales)


def library(z):
    u, ux, uxx = z.T
    return np.column_stack([u, ux, uxx, u*u, u*ux, u*uxx, ux*ux, ux*uxx, uxx*uxx])


def recovered(xi, strong=False):
    a, b, c = (.1, .01, .1) if strong else (.25, .02, .25)
    return bool(abs(xi[4]+1) < a and abs(xi[2]-.02) < b and np.linalg.norm(xi[SPURIOUS]) < c)


def coordinate_check(model, physical, scales):
    with torch.no_grad():
        actual = model(torch.tensor(physical / scales, dtype=torch.float32)).numpy().ravel()
    expected = library(physical.astype(np.float64)) @ coefficients(model, scales)
    error = float(np.max(np.abs(actual - expected)))
    assert np.allclose(actual, expected, atol=2e-5, rtol=2e-5), error
    return error


def field_metrics(model, t, x, refs):
    # Canonical primitive metrics use the dataset-specific physical reference operator.
    def reference_builder(_t, _x, _u, names):
        return {'feature_names': names, 'values': np.column_stack([refs[n].ravel() for n in names])}
    result = evaluate_primitive_feature_metrics(model, t, x, refs['u'].ravel(), PRIMITIVES,
                                               reference_feature_builder=reference_builder)
    fields = evaluate_diagnostic_fields(model, t, x, refs['u'].shape, PRIMITIVES)
    metrics = dict(result['metrics_by_name'])
    metrics['u_t'] = compute_error_metrics(fields['u_t'], refs['u_t'])
    return fields, metrics


def generate_data(out, matched):
    # A=.5, k=1: inviscid characteristic crossing is at t=1/(A*k)=2 > T=1.
    # Positive viscosity is unchanged; quantitative checks below are still required.
    x, _, _, (times, smooth) = solve_burgers(return_history=True, history_every=1,
                                           initial_condition=lambda x: .5*np.sin(x))
    _, _, _, (shock_times, shock) = solve_burgers(seed=0, return_history=True, history_every=1)
    assert np.array_equal(times, shock_times)
    rows, full_fields = [], {}
    for regime, values in [('smooth', smooth), ('shock', shock)]:
        fields, measures = periodic_regularity_metrics(values, x, .02)
        full_fields[regime] = fields
        for i, t in enumerate(times):
            rows.append({'regime': regime, 'time': t, **{k: float(v[i]) for k,v in measures.items()}})
    pd.DataFrame(rows).to_csv(out/'smoothness_over_time.csv', index=False)
    # Historical generator saves initial, steps 1,3,...499,500: 252 observations.
    keep = np.r_[0, np.arange(1, 501, 2), 500]
    xf, tf, uf = x.astype(np.float32), times[keep].astype(np.float32), smooth[keep].astype(np.float32)
    t_flat = np.repeat(tf[:,None], len(xf), axis=1).ravel()
    x_flat = np.repeat(xf[None,:], len(tf), axis=0).ravel()
    assert torch.equal(torch.tensor(t_flat).reshape(-1,1), matched['t'])
    assert torch.equal(torch.tensor(x_flat).reshape(-1,1), matched['x'])
    assert np.array_equal(shock[keep].astype(np.float32).ravel(), matched['u'].numpy().ravel())
    # Refine the verification solver (not the learning data) to bound discretization error.
    xr, _, _, (tr, ur) = solve_burgers(N=512, dt=.001, return_history=True, history_every=1,
                                      initial_condition=lambda x: .5*np.sin(x))
    _, refined = periodic_regularity_metrics(ur, xr, .02)
    difference = ur[::2, ::2] - smooth
    smooth_rows = pd.DataFrame(rows).query("regime == 'smooth'")
    shock_rows = pd.DataFrame(rows).query("regime == 'shock'")
    maxima = {regime: {name: float(df[name].max()) for name in df if name.startswith(('max_', 'spectral_'))}
              for regime, df in [('smooth', smooth_rows), ('shock', shock_rows)]}
    report = {'initial_condition': '0.5*sin(x)', 'amplitude': .5, 'wavenumber': 1.,
              'inviscid_crossing_time': 2., 'horizon': 1., 'nu': .02,
              'solver_steps_checked': len(times), 'Nx': 256, 'Nt': 252, 'observations': 64512,
              'global_maxima': maxima,
              'refined_global_maxima': {k: float(v.max()) for k,v in refined.items()},
              'refinement_max_abs_u': float(np.abs(difference).max()),
              'verification_bounds': {'max_abs_ux': 1.1, 'max_abs_uxx': 2.0, 'refinement_max_abs_u': .001},
              'bounds_basis': 'Predeclared conservative smooth-regime bounds, not recovery-based tuning; A*k*T=.5.',
              'physical_coordinates_match_phase23': True, 'historical_solver_default_unchanged': True}
    assert maxima['smooth']['max_abs_u_x'] < 1.1
    assert maxima['smooth']['max_abs_u_xx'] < 2.
    assert report['refinement_max_abs_u'] < .001
    report['smoothness_pass'] = True
    write_json(out/'smoothness.json', report)
    ref = build_burgers_reference_derivative_grids(u_grid=uf, x_grid=xf, nu=.02)
    refs = {q: ref[q]['physical'] for q in PRIMITIVES+['u_t']}
    np.savez_compressed(out/'reference_fields.npz', x=xf, t=tf, **refs,
                        **{'shock_'+q: full_fields['shock'][q][keep] for q in refs})
    # Exact-reference coefficient-space identifiability control; no learned target involved.
    theta = library(np.column_stack([refs[n].ravel() for n in PRIMITIVES]))
    xi, _, rank, singular = np.linalg.lstsq(theta, refs['u_t'].ravel(), rcond=None)
    colnorm = np.linalg.norm(theta, axis=0)
    scaled = theta/colnorm
    excitation = {'rank': int(rank), 'condition_number': float(singular[0]/singular[-1]),
                  'column_normalized_condition_number': float(np.linalg.cond(scaled)),
                  'coefficients': dict(zip(NAMES,xi.tolist())), 'coefficient_error': float(np.linalg.norm(xi-TRUTH)),
                  'reference_rms': {q:float(np.sqrt(np.mean(refs[q]**2))) for q in refs}}
    assert np.linalg.norm(xi-TRUTH) < 1e-6
    write_json(out/'reference_identifiability.json',excitation)
    return t_flat, x_flat, uf, refs


def estimate_scales(out, t, x, u):
    torch.manual_seed(0); np.random.seed(0)
    model = surrogate()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(1,2501):
        idx = torch.randint(0,len(t),(4096,))
        loss = nn.functional.mse_loss(model(t[idx],x[idx]),u[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            with torch.no_grad(): value = float(nn.functional.mse_loss(model(t,x),u))
            history.append({'stage':'Adam','step':step,'data_mse':value})
            print(f'scale estimator {step}/2500: data MSE {value:.6g}',flush=True)
    opt = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=200,history_size=50,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss=nn.functional.mse_loss(model(t,x),u); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): value=float(nn.functional.mse_loss(model(t,x),u))
    history.append({'stage':'LBFGS','step':2700,'data_mse':value})
    pd.DataFrame(history).to_csv(out/'scale_estimator_history.csv',index=False)
    fields=evaluate_diagnostic_fields(model,t.numpy().ravel(),x.numpy().ravel(),(252,256),PRIMITIVES)
    scales=rms(fields)
    torch.save({'surrogate':model.state_dict(),'scales':scales,'data_mse':value},out/'scale_estimator.pt')
    print('scale estimator complete:',value,scales,flush=True)
    return scales


def train_joint(out, matched, u, scales):
    model, sym = surrogate(), MinimalSymNet()
    model.load_state_dict(matched['surrogate']); sym.load_state_dict(matched['symnet'])
    opt=torch.optim.Adam(list(model.parameters())+list(sym.parameters()),lr=5e-4)
    t,x=matched['t'],matched['x']; scales_tensor=torch.tensor(scales).detach()
    builder=FeatureTensor(PRIMITIVES,normalize=False)
    history=[]
    for epoch in range(1001):
        ix=matched['batches'][epoch-1] if epoch else matched['batches'][0]
        tb=t[ix].detach().clone().requires_grad_(True); xb=x[ix].detach().clone().requires_grad_(True)
        up=model(tb,xb)
        ut=torch.autograd.grad(up,tb,torch.ones_like(up),create_graph=True,retain_graph=True)[0]
        z=builder.build(up,x=xb).F
        data=nn.functional.mse_loss(up,u[ix]); pde=nn.functional.mse_loss(ut,sym(z/scales_tensor))
        loss=data+.5*pde
        assert torch.isfinite(loss)
        xi=coefficients(sym,scales)
        history.append({'epoch':epoch,'data_loss':float(data.detach()),'pde_loss':float(pde.detach()),
                        'total_loss':float(loss.detach()),**{f'xi_{n}':v for n,v in zip(NAMES,xi)}})
        if epoch in CHECKPOINTS:
            torch.save({'epoch':epoch,'surrogate':model.state_dict(),'symnet':sym.state_dict(),
                        'scales':scales,'timing':'before labeled update'},out/'checkpoints'/f'theta_epoch_{epoch:04d}.pt')
            print(f'joint checkpoint {epoch}: data={float(data.detach()):.6g}, PDE={float(pde.detach()):.6g}',flush=True)
        if epoch==1000: break
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    pd.DataFrame(history).to_csv(out/'joint_history.csv',index=False)


def run_probe(frozen, physical, target, scales, seed, checkpoint, historical_beta):
    torch.manual_seed(seed); sym=MinimalSymNet()
    initial_hash=state_hash(sym.state_dict())
    torch.manual_seed(seed); duplicate=MinimalSymNet()
    assert initial_hash==state_hash(duplicate.state_dict())
    beta=np.array(list(product_term_dict(sym).values()))
    initial_coefficient_error=float(np.max(np.abs(beta-historical_beta)))
    assert np.allclose(beta,historical_beta,rtol=2e-5,atol=2e-7)
    before=state_hash(frozen.state_dict())
    assert all(not p.requires_grad for p in frozen.parameters())
    features=torch.tensor(physical/scales.astype(np.float32),dtype=torch.float32)
    target=torch.tensor(target.reshape(-1,1),dtype=torch.float32)
    opt=torch.optim.Adam(sym.parameters(),lr=1e-2)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in sym.parameters()}
    conversion=coordinate_check(sym,physical,scales)
    history=[]
    for epoch in range(1001):
        loss=nn.functional.mse_loss(sym(features),target)
        assert torch.isfinite(loss)
        xi=coefficients(sym,scales)
        history.append({'checkpoint_epoch':checkpoint,'seed':seed,'epoch':epoch,'pde_loss':float(loss.detach()),
                        'coefficient_error':float(np.linalg.norm(xi-TRUTH)),
                        'spurious_l2':float(np.linalg.norm(xi[SPURIOUS])),
                        **{f'xi_{n}':v for n,v in zip(NAMES,xi)}})
        if epoch==1000: break
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    conversion=max(conversion,coordinate_check(sym,physical,scales))
    assert before==state_hash(frozen.state_dict())
    result={**history[-1], 'loose_success':recovered(xi),'strong_success':recovered(xi,True),
            'symnet_initial_state_sha256':initial_hash,'frozen_theta_sha256':before,
            'frozen_unchanged':True,'initial_beta_max_abs_difference_phase23':initial_coefficient_error,
            'coordinate_max_abs_error':conversion}
    return result,history


def summarize(finals, metrics):
    rows=[]
    for _,m in metrics.iterrows():
        g=finals[finals.checkpoint_epoch==int(m.checkpoint_epoch)]
        rows.append({'theta_checkpoint':int(m.checkpoint_epoch),**{q:float(m[q]) for q in PRIMITIVES+['u_t']},
            'recovered':int(g.loose_success.sum()),'total':len(g),'R_k':float(g.loose_success.mean()),
            'strong_recovered':int(g.strong_success.sum()),'coefficient_error_median':float(g.coefficient_error.median()),
            'coefficient_error_mean':float(g.coefficient_error.mean()),'coefficient_error_std':float(g.coefficient_error.std(ddof=1)),
            'transport_median':float(g['xi_u*u_x'].median()),'diffusion_median':float(g.xi_u_xx.median()),
            'spurious_l2_median':float(g.spurious_l2.median())})
    return pd.DataFrame(rows)


def validate_artifacts(out):
    config=json.loads((out/'config.json').read_text())
    assert json.loads((out/'runtime_compatibility.json').read_text())['pass']
    assert json.loads((out/'smoothness.json').read_text())['smoothness_pass']
    verify_inventory(P23,config['phase23_inventory'])
    assert sha256(P22)==config['phase22_inputs_sha256']
    f=pd.read_csv(out/'per_seed_results.csv'); h=pd.read_csv(out/'per_seed_histories.csv')
    m=pd.read_csv(out/'frozen_surrogate_metrics.csv')
    assert len(f)==125 and len(h)==125125 and len(m)==5
    refs=np.load(out/'reference_fields.npz')
    for checkpoint in CHECKPOINTS:
        fields=np.load(out/f'fields_epoch_{checkpoint:04d}.npz')
        row=m[m.checkpoint_epoch==checkpoint].iloc[0]
        for q in PRIMITIVES+['u_t']:
            value=compute_error_metrics(fields[q],refs[q])['mse']
            assert np.isclose(value,row[q],rtol=2e-5,atol=1e-8)
        assert np.allclose(rms(fields),[row['scale_'+q] for q in PRIMITIVES],rtol=1e-6)
        state=torch.load(out/'checkpoints'/f'theta_epoch_{checkpoint:04d}.pt',weights_only=False)
        assert state['epoch']==checkpoint and state['timing']=='before labeled update'
        assert f[f.checkpoint_epoch==checkpoint].frozen_theta_sha256.eq(state_hash(state['surrogate'])).all()
    assert not f.duplicated(['checkpoint_epoch','seed']).any()
    for checkpoint in CHECKPOINTS:
        for seed in range(25):
            g=h[(h.checkpoint_epoch==checkpoint)&(h.seed==seed)]
            assert g.epoch.tolist()==list(range(1001))
            row=f[(f.checkpoint_epoch==checkpoint)&(f.seed==seed)].iloc[0]
            xi=row[[f'xi_{n}' for n in NAMES]].to_numpy(float)
            assert bool(row.loose_success)==recovered(xi)
            assert bool(row.strong_success)==recovered(xi,True)
            assert np.allclose(g.iloc[-1][['pde_loss','coefficient_error','spurious_l2']].to_numpy(float),row[['pde_loss','coefficient_error','spurious_l2']].to_numpy(float))
            assert np.allclose(g.iloc[-1][[f'xi_{n}' for n in NAMES]].to_numpy(float),xi)
    expected=summarize(f,m); actual=pd.read_csv(out/'recoverability_by_checkpoint.csv')
    assert list(expected)==list(actual) and np.allclose(expected.to_numpy(),actual.to_numpy(),atol=1e-12)
    assert f.frozen_unchanged.all() and f.groupby('seed').symnet_initial_state_sha256.nunique().eq(1).all()
    assert f.symnet_initial_state_sha256.nunique()==25
    assert float(m.filter(like='reload_max_abs').to_numpy().max())==0.
    assert np.isfinite(h.select_dtypes('number').to_numpy()).all()
    return {'all_pass':True,'phase23_artifacts_unchanged':True,'rows':{'outcomes':len(f),'histories':len(h),'checkpoints':len(m)},
            'deterministic_reload':True,'frozen_parameters_unchanged':True,'seed_initialization_matches_phase23_with_float32_tolerance':True,
            'coordinate_conversion_pass':True,'max_coordinate_error':float(f.coordinate_max_abs_error.max()),
            'saved_summary_and_endpoint_consistency':True}


def run(out=OUT):
    out=Path(out); torch.set_num_threads(2)
    if (out/'status.json').exists():
        status=json.loads((out/'status.json').read_text())
        if status['status']=='complete':
            verify_inventory(out,json.loads((out/'manifest.json').read_text()))
            validate_artifacts(out)
            print('Reused validated completed Phase 24 artifacts; no files changed.',flush=True)
            return
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite incomplete artifact directory: {out}')
    out.mkdir(parents=True); (out/'checkpoints').mkdir(); (out/'probes').mkdir()
    matched=torch.load(P22,map_location='cpu',weights_only=False)
    baseline_config=json.loads((P23/'config.json').read_text())
    config={'phase':24,'initial_condition':'0.5*sin(x)','domain':[0.,float(2*np.pi)],'nu':.02,'T':1.,'N':256,'dt':.002,
        'Nt':252,'observations':64512,'checkpoints':CHECKPOINTS,'checkpoint_timing':'before labeled update',
        'joint':{'optimizer':'Adam','lr':5e-4,'epochs':1000,'batch_size':4096,'lambda_data':1.,'lambda_pde':.5,
                 'seed':19219,'batch_seed':19220,'regularization':'none','batch_indexing':'0,0,1,...998; no terminal update'},
        'scale_estimator':{'seed':0,'adam_lr':.001,'adam_steps':2500,'lbfgs_max_iter':200,'batch_size':4096,
                           'weights_transferred':False},
        'probe':{'seeds':list(range(25)),'optimizer':'Adam','lr':.01,'epochs':1000,'objective':'full-grid PDE MSE against frozen surrogate u_t',
                 'loose':[.25,.02,.25],'strong':[.1,.01,.1],'scaling':'full-grid RMS recomputed per checkpoint; floor 1e-12'},
        'architecture':'SirenMLP(64,3,20,1) + original MinimalSymNet','primitive_features':PRIMITIVES,'coefficient_order':NAMES,
        'reference_operators':'centered periodic differences, Burgers RHS; physical coordinates',
        'differences_from_phase23':['sinusoidal initial condition replaces seed-0 random bumps',
            'solution-dependent transferred RMS scales recomputed using identical estimator protocol',
            'PyTorch runtime: '+str(torch.__version__)+' vs '+baseline_config['torch_version'],
            'fresh probe seeds use the same RNG protocol; initial coefficients checked at rtol 2e-5, atol 2e-7 because runtime uniform initialization rounds differently',
            'canonical FeatureTensor evaluation replaces equivalent notebook-local primitive construction; numerical compatibility checked'],
        'torch_version':str(torch.__version__),'threads':2,'device':'cpu','dtype':'float32',
        'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'phase23_inventory':inventory(P23),'phase22_inputs_sha256':sha256(P22)}
    write_json(out/'config.json',config)
    t_flat,x_flat,u_grid,refs=generate_data(out,matched)
    t,x=matched['t'],matched['x']; u=torch.tensor(u_grid.ravel()).reshape(-1,1)
    # Initial tensors and probe seeds must equal the actual saved Phase 23 controls.
    torch.manual_seed(19219); sm,ym=surrogate(),MinimalSymNet()
    init_error=max(float((v-matched['surrogate'][k]).abs().max()) for k,v in sm.state_dict().items())
    assert init_error < 2e-7
    sm.load_state_dict(matched['surrogate']); ym.load_state_dict(matched['symnet'])
    config['runtime_initialization_max_abs_difference']=init_error
    config['joint_initialization']='exact archived tensors loaded; seeded regeneration differs by float32 rounding'
    p23_final=pd.read_csv(P23/'per_seed_results.csv')
    p23_history=pd.read_csv(P23/'per_seed_histories.csv')
    p23_init=p23_history[(p23_history.checkpoint_epoch==0)&(p23_history.epoch==0)].set_index('seed')
    p23_scales=pd.read_csv(P23/'frozen_surrogate_metrics.csv').iloc[0]
    old_scales=np.array([p23_scales['scale_'+q] for q in PRIMITIVES])
    initial_betas={seed:p23_init.loc[seed,[f'xi_{n}' for n in NAMES]].to_numpy(float)*coefficient_scales(old_scales) for seed in range(25)}
    del p23_history
    fields=evaluate_diagnostic_fields(sm,t_flat,x_flat,u_grid.shape,PRIMITIVES)
    initial_rms=rms(fields)
    p23_m=pd.read_csv(P23/'frozen_surrogate_metrics.csv').iloc[0]
    assert np.allclose(initial_rms,[p23_m['scale_'+q] for q in PRIMITIVES],rtol=2e-5)
    np.savez_compressed(out/'initial_fields.npz',**fields)
    from utils.burgers_runtime_validation import validate_runtime
    validate_runtime(out)
    scales=estimate_scales(out,t,x,u)
    config['transferred_scales']=scales.tolist()
    write_json(out/'config.json',config)
    train_joint(out,matched,u,scales)
    final_rows=[]; metric_rows=[]; histories=[]; metric_details={}
    for checkpoint in CHECKPOINTS:
        path=out/'checkpoints'/f'theta_epoch_{checkpoint:04d}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        frozen=surrogate(); frozen.load_state_dict(saved['surrogate']); frozen.eval()
        for p in frozen.parameters(): p.requires_grad_(False)
        fields,metrics=field_metrics(frozen,t_flat,x_flat,refs)
        reloaded=surrogate(); reloaded.load_state_dict(saved['surrogate'])
        reload_fields=evaluate_diagnostic_fields(reloaded,t_flat,x_flat,u_grid.shape,PRIMITIVES)
        reload_errors={q:float(np.max(np.abs(fields[q]-reload_fields[q]))) for q in fields}
        assert max(reload_errors.values())==0.
        np.savez_compressed(out/f'fields_epoch_{checkpoint:04d}.npz',**fields)
        scales_k=rms(fields)
        metric_rows.append({'checkpoint_epoch':checkpoint,**{q:metrics[q]['mse'] for q in fields},
            **{'scale_'+q:float(v) for q,v in zip(PRIMITIVES,scales_k)},
            **{'reload_max_abs_'+q:v for q,v in reload_errors.items()}})
        metric_details[str(checkpoint)]=metrics
        physical=np.column_stack([fields[q].ravel() for q in PRIMITIVES])
        # Coefficient-space control separates loss minimization from physical correctness.
        xi_ls=np.linalg.lstsq(library(physical.astype(float)),fields['u_t'].ravel(),rcond=None)[0]
        metric_details[str(checkpoint)]['least_squares']={'coefficients':dict(zip(NAMES,xi_ls.tolist())),
            'coefficient_error':float(np.linalg.norm(xi_ls-TRUTH))}
        for seed in range(25):
            result,history=run_probe(frozen,physical,fields['u_t'],scales_k,seed,checkpoint,initial_betas[seed])
            final_rows.append(result); histories.extend(history)
            # Durable per-seed records retained if execution is interrupted.
            pd.DataFrame(history).to_csv(out/'probes'/f'checkpoint_{checkpoint:04d}_seed_{seed:02d}.csv',index=False)
            write_json(out/'probes'/f'checkpoint_{checkpoint:04d}_seed_{seed:02d}.json',result)
            if seed%5==4: print(f'checkpoint {checkpoint}: {seed+1}/25 probes; last loss={result["pde_loss"]:.6g}, loose={result["loose_success"]}',flush=True)
        pd.DataFrame(final_rows).to_csv(out/'per_seed_results.csv',index=False)
    finals=pd.DataFrame(final_rows); metrics=pd.DataFrame(metric_rows)
    pd.DataFrame(histories).to_csv(out/'per_seed_histories.csv',index=False)
    metrics.to_csv(out/'frozen_surrogate_metrics.csv',index=False)
    write_json(out/'derivative_metrics.json',metric_details)
    summary=summarize(finals,metrics); summary.to_csv(out/'recoverability_by_checkpoint.csv',index=False)
    old=summarize(p23_final,pd.read_csv(P23/'frozen_surrogate_metrics.csv'))
    pd.concat([old.assign(regime='shock'),summary.assign(regime='smooth')]).to_csv(out/'phase23_comparison.csv',index=False)
    validation=validate_artifacts(out)
    validation['initial_surrogate_and_symnet_match_phase23']=True
    validation['initial_primitive_rms_matches_phase23']=True
    write_json(out/'validation.json',validation)
    from utils.burgers_control_plotting import save_plots
    save_plots(out)
    write_json(out/'status.json',{'status':'complete','validation':'passed'})
    write_json(out/'manifest.json',inventory(out))
    print(summary.to_string(index=False),flush=True)
