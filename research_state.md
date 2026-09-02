# Research State

## Current objective
Understand why the joint SIREN + EQL system can achieve low internal PDE loss without recovering the true Burgers equation.

## Current dataset
burg_gen

## True governing PDE
u_t = -u*u_x + 0.02*u_xx

## Current fixed modeling choices
- feature normalization = fixed global scaling
- eql_num_layers = 1
- eql_prod_dim = 2
- hidden_omega_0 = 1.0
- first_omega_0 = 20.0
- lam_data = 1.0
- lam_pde = 0.5

## Known findings
- Unnormalized EQL features caused PDE-loss explosion.
- Fixed-global feature scaling stabilized training.
- eql_num_layers=2 could not represent standalone quadratic u*u_x.
- eql_num_layers=1, eql_prod_dim=2 can represent Burgers support.
- Low internal PDE loss has not yet produced correct Burgers coefficient recovery.
- Previous joint run had poor derivative fidelity, especially u_t.
- Ordinary auxiliary L1 losses have already been removed from the active experiment setup.

## Current open question
Does removing TV and EQL sparsity allow the surrogate and quadratic EQL pathway to fit Burgers structure more faithfully?

## Latest run
Current best/relevant completed run: `run_results/eql_joint_training/burg_gen/seed_000_no_tv_no_sparse_20260902T000000Z_retry1`

Exact config changes from the fixed-global degree-2 baseline:
- `tv_lambda: 1e-6 -> 0.0`
- `lam_sparse_eql: 1e-5 -> 0.0`
- all other settings kept at the prior fixed-global `eql_num_layers=1`, `eql_prod_dim=2` baseline

Main metrics:
- final_total_loss = 0.018574526710879235 at epoch 599
- minimum_total_loss = 0.01845921298104619 at epoch 585
- final_raw_data_mse = 0.01223345236882331 at epoch 599
- minimum_raw_data_mse = 0.009095789195733174 at epoch 0
- final_raw_pde_mse = 0.01268214810757883 at epoch 599
- minimum_raw_pde_mse = 0.012299100602311748 at epoch 585
- final weighted PDE loss = 0.006341074053789415
- minimum weighted PDE loss = 0.006149550301155874 at epoch 585
- full-grid data MSE = 0.011787090217085258

Derivative diagnostics:
- `u` rel_l2 = 0.126750218978615, rmse = 0.10856836655805989
- `u_t` rel_l2 = 0.9933017635022582, rmse = 7.311181096418604
- `u_x` rel_l2 = 0.42343693601543425, rmse = 3.09874185261018
- `u_xx` rel_l2 = 0.7180806118422397, rmse = 356.89760085849497

Coefficient results:
- raw coeff(`u*u_x`) = -0.011966039526585091 versus true `-1.0`
- raw coeff(`u_xx`) = 5.630519373114819e-05 versus true `0.02`
- normalized coeff(`u*u_x`) = -1101.108736141359
- normalized coeff(`u_xx`) = 2.769066333770752
- quadratic product pathway is active, but coefficient recovery remains far from Burgers

Snapshot findings:
- `u` snapshots improved visibly in the interior time range and track coarse profile shape reasonably well.
- largest `u` mismatch remains near the endpoints, especially the latest time slice.
- `u_x` snapshots still miss steeper gradients and show larger slice error around mid-to-late times.
- `u_xx` snapshots remain poor across all selected times and are especially unstable at early and mid times.

Resulting conclusion:
- Removing TV and EQL sparsity improved solution fit and lowered total/data/PDE losses relative to the prior minimal fixed-global baseline.
- The quadratic EQL channel activated strongly, but it did not move the recovered raw Burgers coefficients materially toward the true equation.
- Low internal PDE loss still appears to reflect internal consistency of the learned surrogate-plus-EQL system more than true PDE recovery.

## Next requested experiment
Completed on September 2, 2026:
- `dataset = burg_gen`
- fixed-global feature scaling
- `eql_num_layers = 1`
- `eql_prod_dim = 2`
- `hidden_omega_0 = 1.0`
- `first_omega_0 = 20.0`
- `lam_data = 1.0`
- `lam_pde = 0.5`
- `tv_lambda = 0.0`
- `lam_sparse_eql = 0.0`

Next unresolved research question:
- If removing TV and EQL sparsity improves surrogate fit but leaves `u_t` and `u_xx` fidelity poor and Burgers coefficients far from truth, which remaining baseline choice is driving the internal-consistency failure: the surrogate architecture/training dynamics, the primitive derivative quality, or the fixed-global normalization itself?
