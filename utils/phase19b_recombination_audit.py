"""Independent artifact checks after frozen-head optimization; no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from prog.minimal_symnet import MinimalSymNet
from utils.burgers_recoverability import NAMES, TRUTH, SPURIOUS, coefficients, library
from utils.diagnostic_io import write_json, state_hash, inventory, verify_inventory, sha256

def audit(out,destination=None):
    out=Path(out);dest=out if destination is None else Path(destination)
    cfg=json.loads((out/'config.json').read_text());final=pd.read_csv(out/'per_seed_results.csv')
    scales=np.array(cfg['scales'],dtype=np.float32);rows=[];summary=pd.read_csv(out/'recovery_summary.csv').set_index('condition')
    assert len(final)==100 and not final.duplicated(['condition','seed']).any()
    f={s:np.load(out/f'frozen_{s}.npz') for s in [8000,8750]}
    for s,fields in f.items():
        theta=library(np.column_stack([fields[n].ravel() for n in ['u','u_x','u_xx']]))
        assert np.array_equal(theta,fields['Theta'])
    for c,(s,t) in cfg['pairs'].items():
        for seed in cfg['seeds']:
            h=pd.read_csv(out/'trajectories'/f'{c}_seed_{seed:02d}.csv')
            assert np.array_equal(h.step,np.arange(1001))
            xi=h[['xi_'+n for n in NAMES]].to_numpy();norm=np.linalg.norm(xi[:,SPURIOUS],axis=1)
            assert np.allclose(norm,h.spurious_l2,rtol=1e-10,atol=1e-12)
            assert np.allclose(np.linalg.norm(xi-TRUTH,axis=1),h.coefficient_error,rtol=1e-10,atol=1e-12)
            r=final[(final.condition==c)&(final.seed==seed)].iloc[0]
            for criterion,thresholds in [('loose',(.25,.02,.25)),('strong',(.1,.01,.1))]:
                a,b,d=thresholds;mask=(abs(xi[:,4]+1)<a)&(abs(xi[:,2]-.02)<b)&(norm<d)
                assert np.array_equal(mask,h[criterion]) and bool(mask[-1])==bool(r[criterion])
                first=np.flatnonzero(mask)
                assert (pd.isna(r['first_'+criterion+'_step']) if not len(first) else r['first_'+criterion+'_step']==first[0])
            head=MinimalSymNet();head.load_state_dict(torch.load(out/'heads'/f'{c}_seed_{seed:02d}.pt',weights_only=True))
            assert np.allclose(coefficients(head,scales),xi[-1],rtol=1e-10,atol=1e-12)
            init=torch.load(out/'heads'/f'initial_{seed:02d}.pt',weights_only=True)
            assert state_hash(init)==r.initial_sha256
            head.load_state_dict(init)
            assert np.allclose(coefficients(head,scales),xi[0],rtol=1e-10,atol=1e-12)
            residual=np.mean((f[s]['Theta']@xi[-1]-f[t]['u_t'].ravel())**2)
            assert np.isclose(residual,r.pde_residual,rtol=2e-5,atol=1e-8)
            rows.append({'condition':c,'seed':seed,'last100_coefficient_displacement':float(np.linalg.norm(xi[-1]-xi[-101])),
              'last100_residual_change':float(h.pde_residual.iloc[-1]-h.pde_residual.iloc[-101]),
              'terminal_transport_margin_to_loose':float(.25-abs(xi[-1,4]+1)),
              'terminal_spurious_margin_to_loose':float(.25-norm[-1]),
              'terminal_diffusion_margin_to_loose':float(.02-abs(xi[-1,2]-.02))})
        group=final[final.condition==c]
        for key in ['loose','strong']:assert group[key].sum()==summary.loc[c,key+'_count']
    assert (final.groupby('seed').initial_sha256.nunique()==1).all()
    pd.DataFrame(rows).to_csv(dest/'terminal_diagnostics.csv',index=False)
    prov=json.loads((out/'provenance.json').read_text())
    for root,expected in prov['source_inventories'].items():verify_inventory(root,expected)
    root=Path(__file__).resolve().parents[1]
    assert sha256(root/'notebook/diagnostics/burgers_minimal_discovery_story.ipynb')==prov['notebook_sha256']
    write_json(dest/'independent_audit.json',{'status':'passed','heads_checked':100,'trajectory_rows_checked':100100,
      'paired_initial_state_and_coefficients':True,'physical_library_rebuilt_exactly':True,
      'recovery_flags_and_first_crossings_recomputed':True,'physical_terminal_residuals_recomputed':True,
      'source_inventories_and_notebook_unchanged':True,
      'postprocessing_implementation_sha256':{str(p.relative_to(root)):sha256(p) for p in [Path(__file__),root/'utils/phase19b_recombination_report.py']}})
