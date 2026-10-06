"""Predeclared sustained milestones; no fitted change points or causal labels."""
import json
from pathlib import Path
import numpy as np
from utils.transition_instrumented_io import read_csv,write_csv
from utils.diagnostic_io import write_json


def numeric(path):
    def value(v):
        if v in ['True','False']:return v=='True'
        try:return float(v)
        except ValueError:return v
    return [{k:value(v) for k,v in r.items()} for r in read_csv(path)]


def sustained(rows,predicate):
    for i,r in enumerate(rows):
        window=[s for s in rows[i:] if s['step']<=r['step']+100]
        if window[-1]['step']<r['step']+100:continue
        if all(predicate(s) for s in window):
            return int(r['step']),('left-censored at 7000' if r['step']==7000 else 'sustained at least 100 steps')
    return None,'unresolved / not met'


def events(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text())
    rows=[r for r in numeric(out/'transition_metrics.csv') if r['step']>=7000 and r['step']%25==0]
    gs=[r for r in numeric(out/'gradient_update_metrics.csv') if r['sample']=='fixed' and r['step']>=7000 and r['step']%25==0]
    ls=numeric(out/'ls_trajectory.csv');h=numeric(out/'optimization_history.csv')
    results=[]
    def add(name,criterion,step,status,series='fixed evaluation'):
        results.append({'event':name,'criterion':criterion,'step':step,'status':status,'series':series})
    series=['u_t_rel_l2','u_t_cosine','u_x_rel_l2','u_xx_rel_l2','transport_rel_l2','rhs_rel_l2','rhs_cancellation_ratio','xi_u*u_x','xi_u_xx']
    for key in series:
        a=rows[0][key];b=rows[-1][key]
        for fraction in cfg['event_protocol']['fractional_progress']:
            step,status=sustained(rows,lambda r:(r[key]-a)/(b-a)>=fraction)
            add(key,f'{fraction:.0%} of net 7000-to-10500 change; sustained 100 steps',step,status)
    for key,fn in [('gradient_alignment',lambda r:r['surrogate_data_pde_cosine']>=0),
                   ('weighted_gradient_ratio',lambda r:r['surrogate_weighted_pde_data_ratio']<=1),
                   ('update_aligned_both_losses',lambda r:r['surrogate_update_cos_negative_data']>=0 and r['surrogate_update_cos_negative_pde']>=0)]:
        step,status=sustained(gs,fn);add(key,cfg['event_protocol'][{'gradient_alignment':'gradient_sign','weighted_gradient_ratio':'gradient_ratio','update_aligned_both_losses':'gradient_update_sign'}[key]],step,status,'fixed 4096-point gradients; actual training update')
    for mode in ['learned_learned','learned_reference','reference_learned','true_support_learned','true_support_reference']:
        r=[r for r in ls if r['fit']==mode and r['step']>=7000 and r['step']%25==0]
        for flag in ['loose_recovery','strong_recovery']:
            step,status=sustained(r,lambda r:r[flag]);add(mode+'_'+flag,'original joint coefficient thresholds; sustained 100 steps',step,status,'fixed-grid unconstrained LS')
    for flag in ['loose_recovery','strong_recovery']:
        found=next((int(r['step']) for r in h if r[flag]),None);add('SymNet_'+flag,'original coefficient thresholds; first exact dense crossing',found,'observed','every-step replay')
    rapid=[]
    for i in range(7000,10401):rapid.append({'step':i,'slope':(h[i]['xi_u*u_x']-h[i+100]['xi_u*u_x'])/100})
    step,status=sustained(rapid,lambda r:r['slope']>=.0005);add('rapid_transport','forward 100-step mean transport movement >=0.0005/step, sustained 100 start steps',step,status,'every-step replay')
    # Descriptive fastest smoothed slopes: centered 200-step differences, not onset estimates.
    speeds=[]
    for key in series:
        a=rows[0][key];b=rows[-1][key]
        deriv=[((rows[i+4][key]-rows[i-4][key])/(b-a)/200,int(rows[i]['step'])) for i in range(4,len(rows)-4)]
        speed,step=max(deriv);speeds.append({'quantity':key,'center_step':step,'normalized_progress_per_step':speed,'window_steps':200})
    geo=numeric(out/'geometry_metrics.csv');screens=[]
    for a,b in zip(geo,geo[1:]):
        if a['step']<7000:continue
        change=b['normalized_condition']/a['normalized_condition']-1;angle=b['principal_angle_min']-a['principal_angle_min']
        screens.append({'from_step':int(a['step']),'to_step':int(b['step']),'relative_condition_change':change,
                        'angle_change_degrees':angle,'screen_positive':abs(change)>.2 or abs(angle)>5})
    write_csv(out/'events.csv',results);write_json(out/'events.json',results)
    write_csv(out/'descriptive_peak_rates.csv',speeds);write_json(out/'geometry_change_screen.json',screens)
    return results
