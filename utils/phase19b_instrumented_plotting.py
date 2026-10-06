"""Human-readable synchronized figures from validated instrumented artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.phase19b_instrumented_events import numeric,events


def render(out):
    out=Path(out);events(out)
    d=[r for r in numeric(out/'transition_metrics.csv') if r['step']>=7000]
    g=[r for r in numeric(out/'gradient_update_metrics.csv') if r['step']>=7000 and r['sample']=='fixed']
    ls=numeric(out/'ls_trajectory.csv');geo=[r for r in numeric(out/'geometry_metrics.csv') if r['step']>=7000]
    step=[r['step'] for r in d];gst=[r['step'] for r in g]
    plt.rcParams.update({'font.size':10,'axes.titlesize':11})
    def save(fig,name,title):
        fig.suptitle(title,fontsize=15)
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def marks(ax):
        ax.axvspan(8131,8631,color='gray',alpha=.09)
        for s,col in [(8131,'gray'),(8631,'gray'),(8853,'tab:green'),(10013,'tab:purple')]:ax.axvline(s,color=col,ls=':',lw=1)
        ax.set_xlim(7000,10500);ax.grid(alpha=.2)
    def line(ax,data,x,key,label,**kw):return ax.plot(x,[r[key] for r in data],label=label,**kw)
    fig,axs=plt.subplots(6,1,figsize=(13,16),sharex=True,layout='constrained')
    ax=axs[0];line(ax,d,step,'u_t_rel_l2','u_t relative L2');line(ax,d,step,'u_t_cosine','u_t cosine');ax.set_ylabel('Target fidelity');ax.legend(loc='center left')
    ax.set_title('Gray: fastest archived 8131–8631 window · Green: loose 8853 · Purple: strong 10013')
    ax=axs[1];line(ax,d,step,'rhs_rel_l2','Burgers RHS relative L2');line(ax,d,step,'rhs_cancellation_ratio','Spatial error cancellation ratio');ax.set_ylabel('Spatial compatibility');ax.legend()
    ax=axs[2];line(ax,d,step,'xi_u*u_x','Transport',color='tab:blue');ax.axhline(-1,color='tab:blue',ls='--');ax.set_ylabel('Transport coefficient',color='tab:blue')
    twin=ax.twinx();line(twin,d,step,'xi_u_xx','Diffusion',color='tab:orange');twin.axhline(.02,color='tab:orange',ls='--');twin.set_ylabel('Diffusion coefficient',color='tab:orange')
    ax.legend(loc='center left');twin.legend(loc='center right')
    ax=axs[3]
    for key,label in [('fixed_grid_data_loss','Data'),('fixed_grid_pde_loss','PDE'),('fixed_grid_total_loss','Data + 0.5 PDE')]:line(ax,d,step,key,label)
    ax.set(yscale='log',ylabel='Full-grid MSE');ax.legend()
    ax=axs[4]
    for key,label in [('surrogate_data_pde_cosine','Data/PDE gradients'),('surrogate_update_cos_negative_data','Actual update / −data grad'),('surrogate_update_cos_negative_pde','Actual update / −PDE grad')]:line(ax,g,gst,key,label)
    ax.axhline(0,color='black',lw=.7);ax.set(ylabel='Cosine (fixed sample)',ylim=(-1.05,1.05));ax.legend(ncol=3,fontsize=9)
    ax=axs[5];line(ax,g,gst,'surrogate_weighted_pde_data_ratio','0.5 ||g_PDE|| / ||g_data||',color='tab:blue');ax.set(ylabel='Weighted gradient ratio',yscale='log');ax.axhline(1,color='tab:blue',ls='--')
    twin=ax.twinx();line(twin,g,gst,'surrogate_relative_update_norm','Actual ||Δθ|| / ||θ||',color='tab:orange');twin.set(ylabel='Relative update',yscale='log');ax.legend(loc='upper left');twin.legend(loc='upper right')
    for ax in axs:marks(ax)
    axs[-1].set_xlabel('Completed optimizer steps (state before the next update)')
    save(fig,'figure1_synchronized_transition','Phase 19B transition: representation, coefficients, losses, and updates')
    f=np.load(out/'snapshot_fields.npz');selected=[7500,8000,8500,9000,10000]
    fig,axs=plt.subplots(2,5,figsize=(17,7),sharex=True,sharey=True,layout='constrained')
    for i,q in enumerate(['u_t','rhs']):
        ref=f['reference__u_t'];errors=[]
        for s in selected:
            pred=f[f'{s}__u_t'] if q=='u_t' else -f[f'{s}__u']*f[f'{s}__u_x']+.02*f[f'{s}__u_xx']
            errors.append(pred-ref)
        vmax=max(np.max(np.abs(e)) for e in errors)
        for j,(s,e) in enumerate(zip(selected,errors)):
            im=axs[i,j].pcolormesh(f['x'],f['t'],e,vmin=-vmax,vmax=vmax,cmap='RdBu_r',shading='auto',rasterized=True)
            axs[i,j].set_title(f'{s:,} steps');axs[i,j].set_xlabel('x (physical)')
        axs[i,0].set_ylabel(('u_t' if q=='u_t' else 'Burgers RHS')+' error\nt (physical)');fig.colorbar(im,ax=axs[i,:],label='Prediction − reference')
    save(fig,'figure2_representation_evolution','Target and spatial RHS errors evolve across the recovery transition')
    fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    for ax,key,truth in zip(axs,['xi_u*u_x','xi_u_xx'],[-1,.02]):
        line(ax,d,step,key,'SymNet',color='black',lw=2)
        for label,title in [('learned_learned','Learned features / learned u_t'),('learned_reference','Learned features / reference u_t'),('reference_learned','Reference features / learned u_t'),('true_support_learned','Two learned true terms / learned u_t')]:
            r=[r for r in ls if r['fit']==label and r['step']>=7000];line(ax,r,[v['step'] for v in r],key,title,lw=1.5)
        ax.axhline(truth,color='gray',ls='--',label='Burgers truth');ax.set_ylabel('Physical transport' if truth==-1 else 'Physical diffusion');marks(ax)
    axs[0].legend(ncol=2,fontsize=9);axs[1].set_xlabel('Completed optimizer steps')
    save(fig,'figure3_ls_vs_symnet','Available linear-regression equations versus the trained nonlinear head')
    fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,layout='constrained')
    ax=axs[0,0]
    for k,label in [('surrogate_data_gradient_norm','Data'),('surrogate_pde_gradient_norm','PDE (unweighted)'),('surrogate_total_gradient_norm','Combined')]:line(ax,g,gst,k,label)
    ax.set(yscale='log',ylabel='Fixed-sample gradient norm');ax.legend()
    ax=axs[0,1]
    for q in [10,50,90]:line(ax,g,gst,f'surrogate_inverse_denominator_p{q}',f'p{q}')
    ax.set(yscale='log',ylabel='Adam 1/(√v̂ + ε)');ax.legend()
    ax=axs[1,0];line(ax,g,gst,'surrogate_effective_preconditioning_gain','||m̂/(√v̂+ε)|| / ||m̂||');ax.set_ylabel('Adam effective gain');ax.legend()
    ax=axs[1,1]
    train=[r for r in numeric(out/'gradient_update_metrics.csv') if r['step']>=7000 and r['sample']=='training']
    line(ax,train,gst,'surrogate_momentum_cos_total_gradient','m̂ / actual batch gradient');line(ax,g,gst,'surrogate_update_cos_negative_momentum','Actual update / −m̂');ax.axhline(0,color='gray');ax.set_ylabel('Cosine');ax.legend()
    for ax in axs.ravel():marks(ax);ax.set_xlabel('Completed optimizer steps')
    save(fig,'figure4_adam_geometry','Adam moments and gradient magnitudes: changes versus abrupt events')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    x=[r['step'] for r in geo]
    line(axs[0],geo,x,'raw_condition','Raw physical');axs[0].set(ylabel='Raw condition number',yscale='log')
    line(axs[1],geo,x,'normalized_condition','Unit-column');axs[1].set_ylabel('Unit-column condition number')
    line(axs[2],geo,x,'principal_angle_min','Smallest angle');axs[2].set_ylabel('True/spurious minimum angle (°)')
    for ax in axs:marks(ax);ax.set_xlabel('Completed optimizer steps')
    save(fig,'figure5_feature_geometry','Feature conditioning and separation throughout the transition')
    fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    for key,label in [('u_rel_l2','u'),('u_x_rel_l2','u_x'),('u_xx_rel_l2','u_xx'),('transport_rel_l2','u*u_x'),('weighted_diffusion_rel_l2','0.02 u_xx')]:line(axs[0],d,step,key,label,ls='--' if key=='weighted_diffusion_rel_l2' else '-')
    axs[0].set_ylabel('Relative L2 error');axs[0].legend(ncol=5)
    for key,label in [('transport_error_norm','Transport error'),('weighted_diffusion_error_norm','Weighted diffusion error'),('rhs_error_norm','Combined RHS error'),('closure_norm','Exact-Burgers closure')]:line(axs[1],d,step,key,label)
    axs[1].set_ylabel('Full-grid L2 norm');axs[1].legend(ncol=2)
    for ax in axs:marks(ax)
    axs[1].set_xlabel('Completed optimizer steps')
    save(fig,'figure6_spatial_fidelity','Spatial derivative fidelity and weighted error cancellation')
