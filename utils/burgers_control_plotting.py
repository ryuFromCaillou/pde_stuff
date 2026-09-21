"""Artifact-only plots for the smooth Burgers control."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_plots(out):
    out=Path(out)
    smoothness=pd.read_csv(out/'smoothness_over_time.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for regime,g in smoothness.groupby('regime'):
        for ax,name in zip(axes,['u_x','u_xx']):
            ax.plot(g.time,g['max_abs_'+name],label=regime)
            ax.set(xlabel='physical time',ylabel=f'max |{name}|',yscale='log')
            ax.legend()
    fig.tight_layout(); fig.savefig(out/'smoothness.pdf'); fig.savefig(out/'smoothness.png',dpi=140); plt.close(fig)
    refs=np.load(out/'reference_fields.npz')
    # Learned/reference primitive overlays use the repository's canonical helper.
    import torch
    from utils.burgers_recoverability import surrogate, PRIMITIVES
    from utils.feature_plotting import save_primitive_feature_overlays
    model=surrogate()
    model.load_state_dict(torch.load(out/'checkpoints/theta_epoch_1000.pt',weights_only=False)['surrogate'])
    for parameter in model.parameters(): parameter.requires_grad_(False)
    t_flat=np.repeat(refs['t'][:,None],len(refs['x']),axis=1).ravel()
    x_flat=np.repeat(refs['x'][None,:],len(refs['t']),axis=0).ravel()
    def reference_builder(_t,_x,_u,names):
        return {'feature_names':names,'values':np.column_stack([refs[n].ravel() for n in names])}
    save_primitive_feature_overlays(model,t_flat,x_flat,refs['u'].ravel(),PRIMITIVES,
        output_dir=out/'feature_overlays',reference_feature_builder=reference_builder)
    indices=np.linspace(0,len(refs['t'])-1,5,dtype=int)
    fig,axes=plt.subplots(3,5,figsize=(16,8))
    for row,name in enumerate(['u','u_x','u_xx']):
        for col,i in enumerate(indices):
            ax=axes[row,col]; ax.plot(refs['x'],refs[name][i],label='smooth reference')
            ax.set(title=f't={refs["t"][i]:.3f}',xlabel='physical x',ylabel=name)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'solution_derivative_slices.pdf'); fig.savefig(out/'solution_derivative_slices.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(16,8))
    for row,name in enumerate(['u','u_x','u_xx']):
        for col,i in enumerate(indices):
            ax=axes[row,col]; ax.plot(refs['x'],refs[name][i],label='smooth')
            ax.plot(refs['x'],refs['shock_'+name][i],label='shock-forming')
            ax.set(title=f't={refs["t"][i]:.3f}',xlabel='physical x',ylabel=name)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'reference_regime_slices.pdf'); fig.savefig(out/'reference_regime_slices.png',dpi=140); plt.close(fig)
    comparison=pd.read_csv(out/'phase23_comparison.csv')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for regime,g in comparison.groupby('regime'):
        for ax,key,label in zip(axes.ravel(),['R_k','strong_recovered','u','u_x','u_xx','coefficient_error_median'],
                               ['loose recovery rate','strong recovery count / 25','u MSE','u_x MSE','u_xx MSE','median coefficient error']):
            ax.plot(g.theta_checkpoint,g[key],'o-',label=regime)
            ax.set(xlabel='pre-update joint checkpoint',ylabel=label)
            if key in ['u','u_x','u_xx','coefficient_error_median']: ax.set_yscale('log')
            ax.legend()
    axes[0,0].set_ylim(-.03,1.03); axes[0,1].set_ylim(-.5,25.5)
    fig.tight_layout(); fig.savefig(out/'phase23_comparison.pdf'); fig.savefig(out/'phase23_comparison.png',dpi=140); plt.close(fig)
