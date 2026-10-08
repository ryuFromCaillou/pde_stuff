"""Paul's executable joint path with explicit, mechanically validated parameter L1."""
import argparse
import concurrent.futures
import multiprocessing
import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.featlib import FeatureTensor
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from prog.mlps import SirenMLP

FEATURES = ('u', 'u_x', 'u_xx')
LAMBDAS = (0., 1e-7, 1e-6, 1e-5, 1e-4)
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def siren():
    return SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=1., hidden_omega_0=1.)


def prepare(seed):
    x, _, _, (t, u) = solve_burgers(seed=seed, nu=.02, T=.1, return_history=True)
    x, t, u = (v.astype(np.float32) for v in (x, t, u))
    ti = np.linspace(0, len(t)-1, 10, dtype=int)
    xi = np.linspace(0, len(x)-1, 64, dtype=int)
    def tensors(tt, xx, uu):
        tm, xm = np.meshgrid(tt, xx, indexing='ij')
        return tuple(torch.tensor(v.reshape(-1, 1)) for v in (tm, xm, uu))
    train = tensors(t[ti], x[xi], u[np.ix_(ti, xi)])
    full = tensors(t, x, u)
    torch.manual_seed(seed)
    siren()  # discarded data-only model's initialization
    for _ in range(2500):
        torch.randint(0, 640, (640,))
    MinimalSymNet()  # notebook's exact-Burgers diagnostic head
    model = siren()  # fresh joint model; no pretrained weights
    torch.manual_seed(seed)
    head = MinimalSymNet()
    return model, head, train, full, torch.get_rng_state(), ti, xi


def objective(model, head, batch, lam):
    t, x, truth = batch
    t = t.detach().clone().requires_grad_(True)
    x = x.detach().clone().requires_grad_(True)
    u = model(t, x)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    f = FeatureTensor(FEATURES, normalize=False).build(u, t=t, x=x).F
    data = torch.nn.functional.mse_loss(u, truth)
    pde = torch.nn.functional.mse_loss(ut, head(f))
    raw = sum(p.abs().sum() for p in head.parameters())
    weighted = lam * raw
    return dict(data_loss=data, pde_loss=pde, symnet_l1=raw,
                weighted_l1_loss=weighted, total_loss=data+pde+weighted)


def flatten(values):
    return torch.cat([v.reshape(-1) for v in values])


def validate(model, head, batch):
    results = []
    for dtype in (torch.float32, torch.float64):
        m, h = copy.deepcopy(model).to(dtype), copy.deepcopy(head).to(dtype)
        b = tuple(v.to(dtype) for v in batch)
        hp, mp = list(h.parameters()), list(m.parameters())
        for lam in (*LAMBDAS, .1):
            loss = objective(m, h, b, lam)
            base = loss['data_loss'] + loss['pde_loss']
            expected = base + lam*loss['symnet_l1']
            assert torch.equal(loss['total_loss'], expected)
            g0 = torch.autograd.grad(base, hp+mp, retain_graph=True)
            g1 = torch.autograd.grad(loss['total_loss'], hp+mp, retain_graph=True)
            direct = torch.autograd.grad(loss['weighted_l1_loss'], hp+mp, allow_unused=True)
            assert all(v is None for v in direct[len(hp):])
            diff = flatten(g1[:len(hp)]) - flatten(g0[:len(hp)])
            want = lam * flatten([p.detach().sign() for p in hp])
            tol = 2e-6 if dtype == torch.float32 else 1e-12
            torch.testing.assert_close(diff, want, atol=tol, rtol=tol)
            assert torch.equal(flatten(g0[len(hp):]), flatten(g1[len(hp):]))
            if lam == 0:
                assert torch.equal(base, loss['total_loss'])
                assert torch.equal(flatten(g0), flatten(g1))
            updates = []
            for weight, gradients in ((0., g0), (lam, g1)):
                mc, hc = copy.deepcopy(m), copy.deepcopy(h)
                pars = list(hc.parameters()) + list(mc.parameters())
                opt = torch.optim.Adam(pars, lr=5e-4)
                before = [p.detach().clone() for p in pars]
                objective(mc, hc, b, weight)['total_loss'].backward()
                opt.step()
                for p, old, grad in zip(pars, before, gradients):
                    formula = old - 5e-4 * grad / (grad.abs()+1e-8)
                    torch.testing.assert_close(p, formula, atol=tol, rtol=tol)
                updates.append(flatten([p.detach() for p in hc.parameters()]))
            step_diff = float((updates[1]-updates[0]).norm())
            if lam > 0 and dtype == torch.float64:
                assert step_diff > 0
            results.append(dict(dtype=str(dtype), lambda_l1=lam,
                loss_identity_error=float((loss['total_loss']-expected).detach()),
                symnet_gradient_difference_norm=float(diff.norm()),
                expected_gradient_difference_norm=float(want.norm()),
                gradient_difference_max_error=float((diff-want).abs().max()),
                direct_siren_gradient='absent', matched_siren_gradient_difference_norm=0.,
                symnet_adam_step_difference_norm=step_diff, passed=True))
    # A matched step with populated Adam moments resolves effects hidden by the
    # near sign-only first update. Warm up only the common unregularized state.
    m, h = copy.deepcopy(model), copy.deepcopy(head)
    opt = torch.optim.Adam(list(m.parameters()) + list(h.parameters()), lr=5e-4)
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        objective(m, h, batch, 0.)['total_loss'].backward()
        opt.step()
    matched_updates = []
    for lam in (0., 1e-4):
        mc, hc = copy.deepcopy(m), copy.deepcopy(h)
        pars = list(mc.parameters()) + list(hc.parameters())
        oc = torch.optim.Adam(pars, lr=5e-4)
        oc.load_state_dict(copy.deepcopy(opt.state_dict()))
        oc.zero_grad(set_to_none=True)
        objective(mc, hc, batch, lam)['total_loss'].backward()
        expected = []
        for p in pars:
            state = oc.state[p]
            n = int(state['step']) + 1
            moment = .9 * state['exp_avg'] + .1 * p.grad
            variance = .999 * state['exp_avg_sq'] + .001 * p.grad.square()
            expected.append(p.detach() - 5e-4 * (moment/(1-.9**n)) /
                            ((variance/(1-.999**n)).sqrt()+1e-8))
        oc.step()
        torch.testing.assert_close(flatten(pars), flatten(expected), atol=1e-7, rtol=1e-6)
        matched_updates.append(flatten([p.detach() for p in hc.parameters()]))
    warm_difference = float((matched_updates[1]-matched_updates[0]).norm())
    assert warm_difference > 0
    return dict(passed=True, checks=results,
        matched_adam_after_10_common_steps=dict(lambda_l1=1e-4,
            symnet_step_difference_norm=warm_difference, explicit_adam_formula_passed=True),
        interpretation='Parameter L1 directly affects only SymNet. Coupled future SIREN trajectories can change. First Adam step can round identically in float32; float64 resolves the expected update difference.')


def coefficient_metrics(head):
    c = product_term_dict(head)
    spurious = float(np.linalg.norm([v for k,v in c.items() if k not in ('u*u_x','u_xx')]))
    te, de = abs(c['u*u_x']+1), abs(c['u_xx']-.02)
    loose = te < .25 and de < .02 and spurious < .25
    strong = te < .10 and de < .01 and spurious < .10
    return {**c, 'transport_coefficient': c['u*u_x'], 'diffusion_coefficient': c['u_xx'],
            'spurious_coefficient_norm': spurious, 'loose_recovery': loose,
            'strong_recovery': strong, 'recovery_status': 'strong' if strong else 'loose' if loose else 'none'}


def field_metrics(model, full):
    with torch.no_grad():
        t,x,u = full
        pred = torch.cat([model(tb, xb) for tb,xb in zip(t.split(2048),x.split(2048))])
        return dict(field_mse=float((pred-u).square().mean()), field_relative_l2=float((pred-u).norm()/u.norm()))


def train_one(out, initial_model, initial_head, train, full, rng, lam, steps):
    torch.set_num_threads(1)
    out.mkdir()
    model, head = copy.deepcopy(initial_model), copy.deepcopy(initial_head)
    torch.set_rng_state(rng)
    optimizer = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=5e-4)
    first = {'first_loose_recovery_step': None, 'first_strong_recovery_step': None}
    start = time.monotonic()
    with (out/'history.csv').open('w') as f, (out/'field_metrics.csv').open('w') as ff:
        writer = fw = None
        for step in range(steps+1):
            idx = torch.randint(0, len(train[0]), (640,))
            losses = objective(model, head, tuple(v[idx] for v in train), lam)
            row = {'step': step, **{k:float(v.detach()) for k,v in losses.items()}, **coefficient_metrics(head)}
            if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise RuntimeError(f'Nonfinite trajectory: {lam}, {step}')
            for level in ('loose', 'strong'):
                if row[f'{level}_recovery'] and first[f'first_{level}_recovery_step'] is None:
                    first[f'first_{level}_recovery_step'] = step
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if step % 1000 == 0 or step == steps:
                field = {'step': step, **field_metrics(model, full)}
                if fw is None:
                    fw = csv.DictWriter(ff, fieldnames=list(field)); fw.writeheader()
                fw.writerow(field); f.flush(); ff.flush()
                write_json(out/'progress.json', {'lambda_l1':lam, **row, **field, **first})
                if step % 10000 == 0:
                    torch.save(dict(surrogate=model.state_dict(), symnet=head.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), step=step, next_batch_indices=idx), out/f'state_{step:06d}.pt')
                print(f'lambda={lam:g} step={step} total={row["total_loss"]:.6g} transport={row["transport_coefficient"]:.5g} diffusion={row["diffusion_coefficient"]:.5g} elapsed={time.monotonic()-start:.1f}s', flush=True)
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            losses['total_loss'].backward()
            optimizer.step()
    torch.save(dict(surrogate=model.state_dict(), symnet=head.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), step=steps), out/'final.pt')
    summary = {'lambda_l1':lam, **row, **{k:v for k,v in field.items() if k!='step'}, **first, 'elapsed_seconds':time.monotonic()-start}
    write_json(out/'summary.json', summary)
    return summary


def report(out, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    for name, columns in [('coefficients', list(product_term_dict(MinimalSymNet()))), ('losses', ['data_loss','pde_loss','symnet_l1','weighted_l1_loss','total_loss'])]:
        fig, axes = plt.subplots(3,3 if name=='coefficients' else 2, figsize=(14,10))
        for s in summaries:
            data = pd.read_csv(out/f'lambda_{s["lambda_l1"]:g}'/'history.csv').iloc[::100]
            for ax, col in zip(axes.flat, columns):
                ax.plot(data.step, data[col], label=f'{s["lambda_l1"]:g}')
                ax.set_title(col); ax.set_xlabel('Optimization step')
                ax.set_ylabel('Physical coefficient' if name=='coefficients' else col)
                if name=='losses': ax.set_yscale('symlog', linthresh=1e-8)
        for ax in list(axes.flat)[len(columns):]: ax.set_visible(False)
        axes.flat[0].legend(title='Parameter L1 weight')
        fig.suptitle(f'Paul-path seed 0: {name} versus parameter L1 strength')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(out/f'{name}.{ext}')
        plt.close(fig)
    lines = ['# Paul-path parameter L1 pilot', '', 'Mechanical validation passed. Parameter L1 is on internal factors, not expanded PDE coefficients. Direct SIREN penalty gradients are absent; subsequent coupled trajectories can change.', '', '|lambda|transport|diffusion|spurious L2|field MSE|first loose|first strong|endpoint|', '|---|---|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f'|{s["lambda_l1"]:g}|{s["transport_coefficient"]:.6g}|{s["diffusion_coefficient"]:.6g}|{s["spurious_coefficient_norm"]:.6g}|{s["field_mse"]:.6g}|{s["first_loose_recovery_step"]}|{s["first_strong_recovery_step"]}|{s["recovery_status"]}|')
    lines += ['', 'Single matched seed; first crossings need not persist. Field errors use the full clean generator grid. Loss histories are sampled minibatch losses. See config.json and runs/PAUL_PATH_L1.md for protocol and timing.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1,2,3), default=1)
    args = parser.parse_args()
    if args.steps < 1: parser.error('--steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model,head,train,full,rng,ti,xi = prepare(0)
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['pauls_tweaks_diag_notebook.ipynb','prog/mlps.py','prog/minimal_symnet.py','prog/featlib.py','utils/paul_path_l1.py','Datasets/data/processed/burg_gen/burg_gen.py']}
    config = dict(seed=0, dataset_args=dict(nu=.02,T=.1,N=256,dt=.002), time_indices=ti.tolist(), space_indices=xi.tolist(), architecture='SirenMLP 3x64 omega 1/1; fresh MinimalSymNet', features=FEATURES, normalize=False, lambda_data=1., lambda_pde=1., lambdas=LAMBDAS, lr=5e-4, optimizer='Adam default betas=(.9,.999), eps=1e-8, weight_decay=0', steps=args.steps, batch_size=640, sampling='with replacement; matched RNG per treatment', dtype='float32', device='cpu', threads_per_worker=1, workers=args.workers, torch=torch.__version__, python=platform.python_version(), source_hashes=hashes, recovery=dict(loose=[.25,.02,.25],strong=[.10,.01,.10],order=['transport absolute error','diffusion absolute error','spurious L2'],strict=True), smoke=args.steps!=100000)
    write_json(args.output/'config.json',config)
    torch.save(dict(surrogate=model.state_dict(),symnet=head.state_dict(),train=train,full=full,rng=rng),args.output/'matched_inputs.pt')
    torch.set_rng_state(rng)
    idx = torch.randint(0,640,(640,))
    validation = validate(model,head,tuple(v[idx] for v in train))
    write_json(args.output/'validation.json',validation)
    print('Mechanical validation passed',flush=True)
    if args.validate_only: return
    summaries = []
    if args.workers == 1:
        for lam in LAMBDAS:
            summaries.append(train_one(args.output/f'lambda_{lam:g}',model,head,train,full,rng,lam,args.steps))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = [pool.submit(train_one,args.output/f'lambda_{lam:g}',model,head,train,full,rng,lam,args.steps) for lam in LAMBDAS]
            for job in jobs:
                summaries.append(job.result())
    report(args.output,summaries)
    assert hashlib.sha256((ROOT/'pauls_tweaks_diag_notebook.ipynb').read_bytes()).hexdigest()==hashes['pauls_tweaks_diag_notebook.ipynb']
