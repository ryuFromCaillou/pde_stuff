"""Numerical compatibility replay against read-only Phase 23 artifacts."""
import ast,json
import numpy as np
import pandas as pd
import torch
from utils.burgers_recoverability import (ROOT, P22, P23, OUT, PRIMITIVES, NAMES, MinimalSymNet,
    surrogate, rms, coefficient_scales, evaluate_diagnostic_fields, run_probe, state_hash)
from torch import nn
from utils.diagnostic_io import write_json


def validate_runtime(out=OUT):
    torch.set_num_threads(2)
    nb=json.loads((ROOT/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb').read_text())
    # Compare extracted reusable model against original notebook definition for all probe seeds.
    scope={'torch':torch,'nn':nn}
    node=ast.parse(''.join(nb['cells'][15]['source'])).body[0]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook MinimalSymNet>','exec'),scope)
    for seed in range(25):
     torch.manual_seed(seed); a=scope['MinimalSymNet']()
     torch.manual_seed(seed); b=MinimalSymNet()
     assert state_hash(a.state_dict())==state_hash(b.state_dict())
    # Replay one historical terminal-state probe to quantify environment/evaluation differences.
    matched=torch.load(P22,weights_only=False)
    sm=surrogate();sm.load_state_dict(torch.load(P23/'checkpoints/theta_epoch_1000.pt',weights_only=False)['surrogate'])
    for p in sm.parameters():p.requires_grad_(False)
    t=matched['t'].numpy().ravel();x=matched['x'].numpy().ravel()
    f=evaluate_diagnostic_fields(sm,t,x,(252,256),PRIMITIVES)
    old_metric=pd.read_csv(P23/'frozen_surrogate_metrics.csv').query('checkpoint_epoch == 1000').iloc[0]
    sc=rms(f); old_sc=np.array([old_metric['scale_'+q] for q in PRIMITIVES])
    assert np.allclose(sc,old_sc,rtol=2e-5)
    first=pd.read_csv(P23/'per_seed_histories.csv').query('checkpoint_epoch == 1000 and seed == 0 and epoch == 0').iloc[0]
    beta=first[[f'xi_{n}' for n in NAMES]].to_numpy(float)*coefficient_scales(old_sc)
    physical=np.column_stack([f[q].ravel() for q in PRIMITIVES])
    result,_=run_probe(sm,physical,f['u_t'],sc,0,1000,beta)
    old=pd.read_csv(P23/'per_seed_results.csv').query('checkpoint_epoch == 1000 and seed == 0').iloc[0]
    old_xi=old[[f'xi_{n}' for n in NAMES]].to_numpy(float)
    new_xi=np.array([result[f'xi_{n}'] for n in NAMES])
    report={'original_notebook_model_matches_extracted_model_all_seeds':True,'checkpoint':1000,'seed':0,
     'historical_final_loss':float(old.final_pde_loss),'replayed_final_loss':result['pde_loss'],
     'max_abs_coefficient_difference':float(np.abs(new_xi-old_xi).max()),
     'historical_loose_success':bool(old.loose_success),'replayed_loose_success':result['loose_success'],
     'historical_strong_success':bool(old.strong_success),'replayed_strong_success':result['strong_success'],
     'scope':'One terminal-state seed replay, not a complete historical experiment rerun; existing artifacts read only.'}
    assert np.allclose(new_xi,old_xi,rtol=1e-3,atol=1e-4),report
    assert np.isclose(result['pde_loss'],old.final_pde_loss,rtol=1e-3,atol=1e-6),report
    report['pass']=True
    write_json(out/'runtime_compatibility.json',report)
    print(json.dumps(report,indent=2))
