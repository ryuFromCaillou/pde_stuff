"""Artifact-only visualization for the Phase 19B transition audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def render(out):
    out=Path(out);m=json.loads((out/'metrics.json').read_text());f=np.load(out/'fields.npz')
    names=m['names'];states=['reference','5000','10000'];labels=['Reference','5k','10k'];sp=m['spurious_indices'];ti=m['true_indices']
    plt.rcParams.update({'font.size':10,'axes.titlesize':11})
    def save(fig,name,title):
        fig.suptitle(title,fontsize=15)
        fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for j,q in enumerate(['u','u_x','u_xx','u_t']):
        es=[f[s+'__'+q]-f['reference__'+q] for s in states[1:]];v=max(np.max(np.abs(e)) for e in es)
        for i,e in enumerate(es):
            im=axs[i,j].pcolormesh(f['x'],f['t'],e,cmap='RdBu_r',vmin=-v,vmax=v,shading='auto',rasterized=True)
            axs[i,j].set(title=f'{labels[i+1]}: {q} error',xlabel='x (physical coordinate)',ylabel='t (physical time)')
        fig.colorbar(im,ax=axs[:,j],label='prediction − reference')
    save(fig,'figure1_error_maps','Field and derivative errors: shared limits for 5k and 10k')
    for raw in [False,True]:
        fig,axs=plt.subplots(1,3,figsize=(17,6),layout='constrained')
        for ax,s,l in zip(axs,states,labels):
            a=np.array(m['geometry'][s]['raw_gram_mean' if raw else 'normalized_gram'])
            if raw:im=ax.imshow(a,cmap='RdBu_r',norm=SymLogNorm(linthresh=1,vmin=-1e9,vmax=1e9))
            else:im=ax.imshow(a,cmap='RdBu_r',vmin=-1,vmax=1)
            ax.set(xticks=range(9),yticks=range(9),xticklabels=names,yticklabels=names,title=l)
            ax.tick_params(axis='x',rotation=65)
        fig.colorbar(im,ax=axs,label='mean physical column products (mixed units; symmetric log)' if raw else 'Uncentered cosine / unit-column Gram entry',shrink=.75)
        save(fig,'figure2_raw_gram' if raw else 'figure2_normalized_gram','Physical feature Gram matrices: raw scale' if raw else 'Normalized feature geometry: false alternatives persist')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,kind in zip(axs,['normalized','raw']):
        for s,l in zip(states,labels):
            g=m['geometry'][s][kind];ax.semilogy(range(1,10),g['singular_values'],'o-',label=f'{l}: κ={g["condition"]:.3g}')
        ax.set(xlabel='Singular value index',ylabel='Singular value',title='Unit-L2 columns' if kind=='normalized' else 'Physical columns (mixed units)');ax.legend();ax.grid(alpha=.2)
    save(fig,'figure3_singular_values','Library spectra: normalization separates scale from collinearity')
    fig,axs=plt.subplots(1,3,figsize=(12,6),layout='constrained')
    for ax,s,l in zip(axs,states,labels):
        a=np.array(m['geometry'][s]['normalized_gram'])[np.ix_(sp,ti)]
        im=ax.imshow(a,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        for i in range(7):
            for j in range(2):ax.text(j,i,f'{a[i,j]:+.2f}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=[names[i] for i in ti],yticks=range(7),yticklabels=[names[i] for i in sp],title=l)
    fig.colorbar(im,ax=axs,label='Signed uncentered cosine',shrink=.8)
    save(fig,'figure4_true_spurious_alignment','True-term versus spurious-feature alignment does not disappear')
    fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for s,l,offset in [('5000','5k',-.18),('10000','10k',.18)]:
        c=next(r['coefficients'] for r in m['fits'] if r['state']==s and r['target']=='predicted' and r['support']=='full')
        for ax,idx in [(axs[0],range(9)),(axs[1],sp)]:ax.bar(np.arange(len(idx))+offset,np.array(c)[list(idx)],width=.34,label=l)
    for ax,idx in [(axs[0],list(range(9))),(axs[1],sp)]:
        ax.scatter(range(len(idx)),np.array(m['truth'])[idx],marker='_',s=300,color='black',label='Burgers truth',zorder=4)
        ax.set(xticks=range(len(idx)),xticklabels=[names[i] for i in idx],ylabel='Physical coefficient (term-specific units)');ax.axhline(0,color='gray',lw=.7);ax.legend()
    axs[0].set_title('All nine coefficients: full-library least squares on predicted u_t')
    axs[1].set_title('Expanded view of the seven spurious coefficients')
    save(fig,'figure5_ls_coefficients','Coefficient-space regression changes from weak transport to Burgers-like support')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ens=['e_t','e_transport','e_diffusion','closure_error'];spaces=['predicted_true','predicted_spurious']
    for ax,s,l in zip(axs,states[1:],labels[1:]):
        a=np.array([[next(p['energy_fraction'] for p in m['projections'] if p['state']==s and p['error']==e and p['space']==space) for space in spaces] for e in ens])
        im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i in range(4):
            for j in range(2):ax.text(j,i,f'{a[i,j]:.1%}',ha='center',va='center',color='white' if a[i,j]<.5 else 'black')
        ax.set(xticks=[0,1],xticklabels=['True-term span','Spurious-term span'],yticks=range(4),yticklabels=ens,title=l)
    fig.colorbar(im,ax=axs,label='Fraction of error energy projected (not additive)',shrink=.8)
    save(fig,'figure6_error_projections','Remaining errors become more aligned with spurious directions')
    fig,axs=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for j,time in enumerate([.25,.65,1.]):
        idx=int(np.argmin(abs(f['t']-time)))
        for i,q in enumerate(['u','u_x','u_xx']):
            ax=axs[i,j]
            for s,l in zip(states,labels):ax.plot(f['x'],f[s+'__'+q][idx],label=l,lw=1.4)
            ax.set(xlabel='x (physical coordinate)',ylabel=q,title=f't={f["t"][idx]:.3f}');ax.grid(alpha=.2)
    axs[0,0].legend()
    save(fig,'figure7_physical_snapshots','Localized steep-gradient discrepancies survive coefficient recovery')
