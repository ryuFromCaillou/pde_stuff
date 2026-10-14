"""Read-only validation of completed physical-head artifacts; writes to a fresh audit folder."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.physics_informed_burgers import hashes, NOTEBOOKS, initialize, coefficients, write_json


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    config=json.loads((source/'config.json').read_text())
    reference=np.load(source/'reference.npz')
    summaries=[]
    results=[]
    diagnostics=[]
    expected_steps=config['steps']
    for seed in config['seeds']:
        folder=source/f'seed_{seed}'
        if not (folder/'summary.json').exists():
            results.append(dict(seed=seed,complete=False))
            continue
        summary=json.loads((folder/'summary.json').read_text())
        h=pd.read_csv(folder/'history.csv',float_precision='round_trip')
        assert len(h)==expected_steps+1
        np.testing.assert_array_equal(h.step,np.arange(expected_steps+1))
        assert np.isfinite(h.select_dtypes(include='number')).all().all()
        np.testing.assert_allclose(h.total_loss,h.data_loss+h.pde_loss,rtol=2e-7,atol=1e-10)
        np.testing.assert_allclose(h.transport_abs_error,abs(h.transport+1),rtol=1e-6,atol=1e-12)
        np.testing.assert_allclose(h.diffusion_abs_error,abs(h.diffusion-.02),rtol=1e-6,atol=1e-12)
        for level,thresholds in [('loose',(.25,.02)),('strong',(.1,.01))]:
            mask=(abs(h.transport+1)<thresholds[0]) & (abs(h.diffusion-.02)<thresholds[1])
            np.testing.assert_array_equal(h[level],mask)
            first=int(h.loc[mask,'step'].iloc[0]) if mask.any() else None
            assert first==summary[f'first_{level}_step']
            assert bool(mask.iloc[-1])==summary[level]
        checkpoint_metrics=json.loads((folder/'checkpoint_metrics.json').read_text())
        discrepancy=0.
        for m in checkpoint_metrics:
            step=m['step']; row=h.iloc[step]
            fields=np.load(folder/f'fields_{step:06d}.npz')
            state=torch.load(folder/f'state_{step:06d}.pt',weights_only=False,map_location='cpu')
            np.testing.assert_array_equal(state['head']['xi'].numpy(),[row.transport,row.diffusion])
            assert state['step']==step
            residual=fields['u_t'].astype(float)-row.transport*fields['u'].astype(float)*fields['u_x'].astype(float)-row.diffusion*fields['u_xx'].astype(float)
            np.testing.assert_allclose(fields['residual'],residual,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(np.mean(residual**2),m['independent_pde_mse'],rtol=1e-10,atol=1e-12)
            for name in ('u','u_x','u_xx','u_t'):
                ref=reference[f'ref_{name}']; pred=fields[name].astype(float)
                key=name.replace('_','')
                mse=float(np.mean((pred-ref)**2)); rel=float(np.linalg.norm(pred-ref)/(np.linalg.norm(ref)+1e-12))
                np.testing.assert_allclose([m[f'{key}_mse'],m[f'{key}_rel_l2']],[mse,rel],rtol=1e-9,atol=1e-12)
            discrepancy=max(discrepancy,m['independent_residual_check_abs_error'])
        assert all(summary[k]==h.iloc[-1][k] for k in ('loose','strong'))
        for k in ('transport','diffusion','data_loss','pde_loss','total_loss'):
            np.testing.assert_allclose(summary[k],h.iloc[-1][k],rtol=1e-12,atol=1e-15)
        # Rebuild initial seed and verify no pretrained weights entered training.
        model,head=initialize(seed)
        initial=torch.load(folder/'state_000000.pt',weights_only=False,map_location='cpu')
        assert all(torch.equal(v,initial['surrogate'][k]) for k,v in model.state_dict().items())
        assert torch.equal(head.xi,initial['head']['xi'])
        last=h[h.step>=max(0,expected_steps-20000)]
        final_fields=np.load(folder/f'fields_{expected_steps:06d}.npz')
        truth_residual=(final_fields['u_t'].astype(float)
            +final_fields['u'].astype(float)*final_fields['u_x'].astype(float)
            -.02*final_fields['u_xx'].astype(float))
        diagnostics.append(dict(seed=seed,final_status=summary['status'],
            transport_abs_error=summary['transport_abs_error'],diffusion_abs_error=summary['diffusion_abs_error'],
            late_transport_change=float(last.transport.iloc[-1]-last.transport.iloc[0]),
            late_diffusion_change=float(last.diffusion.iloc[-1]-last.diffusion.iloc[0]),
            late_strong_fraction=float(last.strong.mean()),late_loose_fraction=float(last.loose.mean()),
            learned_coefficient_residual_mse=summary['independent_pde_mse'],
            true_coefficient_on_learned_fields_residual_mse=float(np.mean(truth_residual**2)),
            u_relative_error=summary['u_rel_l2'],ut_relative_error=summary['ut_rel_l2'],
            ux_relative_error=summary['ux_rel_l2'],uxx_relative_error=summary['uxx_rel_l2']))
        summaries.append(summary)
        results.append(dict(seed=seed,complete=True,history_rows=len(h),checkpoints=len(checkpoint_metrics),
                            maximum_independent_residual_check_abs_error=discrepancy,passed=True))
    unchanged=hashes(NOTEBOOKS)=={k:config['source_hashes'][k] for k in NOTEBOOKS}
    assert unchanged
    source_match=hashes(config['source_hashes'])==config['source_hashes']
    assert source_match
    reuse=source/'reused_artifacts.json'
    if reuse.exists():
        for relative,digest in json.loads(reuse.read_text())['hashes'].items():
            assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==digest
    pd.DataFrame(diagnostics).to_csv(output/'endpoint_diagnostics.csv',index=False)
    lines=['# Clean-baseline endpoint investigation', '',
        'Post-hoc descriptive checks; no additional optimization or least-squares extraction.',
        'Late changes and threshold occupancy use the last 20000 optimizer steps. True-coefficient residuals substitute the physical targets into saved learned fields; they do not fix coefficients during training.', '']
    for d in diagnostics:
        lines.append(f"Seed {d['seed']}: endpoint {d['final_status']}; absolute coefficient errors ({d['transport_abs_error']:.6g}, {d['diffusion_abs_error']:.6g}); late coefficient changes ({d['late_transport_change']:.6g}, {d['late_diffusion_change']:.6g}); late strong fraction {d['late_strong_fraction']:.4f}.")
    if any(d['final_status']!='strong' for d in diagnostics):
        lines += ['', 'Strong clean recovery is not uniform within the declared horizon. The saved late changes distinguish continued progress from an approximately stationary endpoint; they do not establish eventual recovery. Compare remaining temporal/spatial derivative errors and the true-coefficient residual with the learned-head residual in endpoint_diagnostics.csv. Low learned residual alone cannot certify the physical coefficients. Do not proceed to noise or sparsity or tune failed seeds independently. A matched all-seed horizon or objective investigation requires a separately reviewed protocol.']
    lines += ['', 'These observations cannot identify a unique optimization failure mechanism. Five repeats estimate variability only on one fixed numerical initial condition and grid.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    report=dict(passed=all(r.get('passed',False) for r in results),seeds=results,
                notebooks_unchanged=unchanged,implementation_hashes_match=source_match,
                scope='Read-only scalar, checkpoint, cached-field, reference, initialization, recovery-time and provenance audit.')
    write_json(output/'validation.json',report)
    return report
