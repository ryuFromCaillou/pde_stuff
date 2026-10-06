"""Independent numerical consistency checks on cached transition artifacts."""
import json
from pathlib import Path
import numpy as np
import torch
from prog.minimal_symnet import MinimalSymNet
from utils.burgers_recoverability import library, PRIMITIVES
from utils.diagnostic_io import verify_inventory, write_json
from utils.phase19b_transition import SOURCE

def validate(out):
    out=Path(out);m=json.loads((out/'metrics.json').read_text());v=json.loads((out/'validation.json').read_text());f=np.load(out/'fields.npz')
    scales=np.array(json.loads((SOURCE/'config.json').read_text())['scales'])
    results=[]
    for s in ['5000','10000','20000']:
        z=np.column_stack([f[s+'__'+k].ravel() for k in PRIMITIVES]);a=library(z);y=f[s+'__u_t'].ravel()
        c=np.linalg.lstsq(a,y,rcond=1e-12)[0]
        full=next(r for r in m['fits'] if r['state']==s and r['target']=='predicted' and r['support']=='full')
        delta=float(np.max(abs(c-np.array(full['coefficients']))));assert delta<1e-8
        true=next(r for r in m['fits'] if r['state']==s and r['target']=='predicted' and r['support']=='true_only')
        gain=full['incremental_spurious_fraction_of_true_residual']*true['residual_mse']
        assert np.isclose(gain,full['true_minus_full_mse'],rtol=1e-10,atol=1e-12)
        state=torch.load(SOURCE/f'checkpoint_{int(s):06d}.pt',map_location='cpu',weights_only=False)
        sym=MinimalSymNet().double();sym.load_state_dict(state['symnet'])
        with torch.no_grad(): actual=sym(torch.tensor(z/scales,dtype=torch.float64)).numpy().ravel()
        expected=a@np.array(m['trained_heads'][s]);err=float(np.max(abs(actual-expected)))
        assert np.allclose(actual,expected,rtol=2e-6,atol=2e-6),err
        results.append({'state':s,'raw_vs_normalized_coeff_max_abs_difference':delta,'symbolic_expansion_max_abs_difference':err,
                        'incremental_spurious_identity_verified':True})
    verify_inventory(SOURCE,v['source_inventory'])
    v['additional_checks']=results;v['source_unchanged']=True
    write_json(out/'validation.json',v)
    print('Independent cached-field checks passed',flush=True)
