"""Build a small presentation snapshot from the local, still-running pilot.

Run from repository root; pass a fresh destination to preserve older snapshots.
"""
import argparse
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
if (out/'README.md').exists():
    raise SystemExit('Snapshot already exists; choose a fresh destination.')
source = Path('run_results/paul_path_l1_pilot_validated')
rows, histories, fields = [], [], []
for lam in [0., 1e-7, 1e-6, 1e-5, 1e-4]:
    folder = source/f'lambda_{lam:g}'
    if not (folder/'progress.json').exists():
        rows.append({'lambda_l1': lam, 'run_status': 'queued'})
        continue
    progress = json.loads((folder/'progress.json').read_text())
    step = progress['step']
    # Only complete lines at or before the last fully reported state.
    raw = (folder/'history.csv').read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]
    history = pd.read_csv(io.BytesIO(raw))
    history = history[(history.step <= step) & (history.step % 1000 == 0)].copy()
    history.insert(0, 'lambda_l1', lam)
    histories.append(history)
    field = pd.read_csv(folder/'field_metrics.csv')
    field = field[field.step <= step].copy()
    field.insert(0, 'lambda_l1', lam)
    fields.append(field)
    rows.append({**progress, 'run_status': 'complete' if (folder/'summary.json').exists() else 'running'})
pd.DataFrame(rows).to_csv(out/'progress_summary.csv', index=False)
history = pd.concat(histories, ignore_index=True)
field = pd.concat(fields, ignore_index=True)
history.to_csv(out/'trajectory_every_1000_steps.csv', index=False)
field.to_csv(out/'field_metrics.csv', index=False)
for name in ['validation.json', 'config.json']:
    shutil.copy(source/name, out/name)
shutil.copy(Path('run_results/paul_path_l1_mse_smoke/original_objective_validation.json'), out/'original_objective_validation.json')
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for lam, group in history.groupby('lambda_l1'):
    for ax, col in zip(axes.flat, ['transport_coefficient', 'diffusion_coefficient', 'spurious_coefficient_norm']):
        ax.plot(group.step, group[col], label=f'{lam:g}')
        ax.set_title(col.replace('_',' ')); ax.set_xlabel('Optimization step')
    fg = field[field.lambda_l1 == lam]
    axes[1,1].semilogy(fg.step, fg.field_relative_l2, label=f'{lam:g}')
axes[0,0].axhline(-1, color='black', ls=':', label='Target')
axes[0,1].axhline(.02, color='black', ls=':')
axes[1,1].set_title('Full-grid field relative L2 error'); axes[1,1].set_xlabel('Optimization step')
for ax in axes.flat: ax.legend(title='Parameter L1 weight'); ax.grid(alpha=.2)
fig.suptitle('Paul-path L1 pilot: preliminary trajectories (seed 0, runs incomplete)')
fig.tight_layout()
for ext in ['png','pdf']: fig.savefig(out/f'preliminary_trajectories.{ext}', dpi=140)
plt.close(fig)
now = datetime.now(timezone.utc).isoformat()
lines = ['# Paul-path parameter L1: presentation snapshot', '', f'Snapshot UTC: {now}', '',
'**Preliminary: the 100,000-step pilot is still running.** This is a compact presentation snapshot, not the final L1-strength comparison. Only three conditions have started; lambda 1e-5 and 1e-4 are queued.', '',
'## What is established', '',
'- The runner includes parameter L1 in the joint objective. Zero lambda matches the unregularized objective and gradients exactly.',
'- The loss and gradients match the original notebook loss code after including its already computed L1 term.',
'- At lambda 1e-4, the SymNet gradient difference norm is about 3.16225e-4. Direct SIREN penalty gradients are absent; future coupled trajectories may change.',
'- A matched float32 Adam step after ten common unregularized steps differs by 2.61264e-6 in SymNet parameters. First-step float32 rounding is covered separately by float64 checks.',
'- This penalizes internal factorized parameters, **not expanded physical PDE coefficients**.', '',
'## Current progress', '', '|lambda|step|transport (target -1)|diffusion (target .02)|spurious L2|recovery|', '|---|---|---|---|---|---|']
for row in rows:
    if row['run_status']=='queued':
        lines.append(f"|{row['lambda_l1']:g}|queued|—|—|—|—|")
    else:
        lines.append(f"|{row['lambda_l1']:g}|{row['step']}|{row['transport_coefficient']:.5g}|{row['diffusion_coefficient']:.5g}|{row['spurious_coefficient_norm']:.5g}|{row['recovery_status']}|")
lines += ['', '![Preliminary trajectories](preliminary_trajectories.png)', '',
'Files: `progress_summary.csv` contains current metrics and first recovery crossings; `trajectory_every_1000_steps.csv` contains all nine physical coefficients and five loss components; `field_metrics.csv` contains full-grid field errors. JSON files record mechanical validation and configuration.', '',
'Full histories, checkpoints, logs, older sparsity-sweep artifacts, and smoke plots are excluded to keep this presentation small. The complete artifacts remain local. Reproduce training with `runs/run_paul_path_l1.py`; protocol: `runs/PAUL_PATH_L1.md`. Use `build_snapshot.py --output <fresh-directory>` from the repository root to build a later snapshot.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(out)
