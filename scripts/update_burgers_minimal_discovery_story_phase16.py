from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebook/diagnostics/burgers_minimal_discovery_story.ipynb")


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


PHASE16_MARKDOWN = """
## 16. Controlled Initialization and Convergence Map

Question answered here: on the exact same frozen Phase 9 surrogate derivative field and the original `MinimalSymNet`, how systematically does convergence depend on deliberately chosen initialization families rather than isolated random seeds?

This phase preserves and reuses:

- `theta_good`: the earlier Burgers-like frozen-SymNet point reached from the hand-initialized good basin,
- `theta_bad`: the known bad random-final frozen-SymNet point from Phase 9 / Phase 12,
- `xi_true`: the true Burgers coefficient vector,
- `xi_LS`: the Phase 11 least-squares coefficient vector on the frozen surrogate field.

The only intended variable here is initialization. The derivative data, feature library, optimizer, learning rate, epochs, and architecture stay fixed.
"""


PHASE16_CODE = """
theta_good = theta_ref.clone().detach()
xi_true = phase14_true_coeff_vector.copy()
xi_LS = phase15_xi_ls.copy()
phase16_feature_names = list(phase12_coeff_names)
phase16_checkpoints = [0, 10, 100, 500, 1000]
phase16_good_pde_threshold = phase15_return_pde_threshold
phase16_good_xi_threshold = phase15_return_xi_threshold


def phase16_assign_theta(model, theta_flat):
    theta_tensor = torch.as_tensor(theta_flat, dtype=torch.float32, device=model.left.weight.device)
    with torch.no_grad():
        model.left.weight.copy_(theta_tensor[0:3].reshape(1, 3))
        model.right.weight.copy_(theta_tensor[3:6].reshape(1, 3))
        model.linear.weight.copy_(theta_tensor[6:9].reshape(1, 3))
        model.product_readout.weight.copy_(theta_tensor[9].reshape(1, 1))


def phase16_theta_from_equation(product_coeff, diffusion_coeff):
    return torch.tensor(
        [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, float(diffusion_coeff),
         float(product_coeff)],
        dtype=torch.float64,
    )


def phase16_scale_product_path(theta_base, scale):
    theta_scaled = theta_base.clone().detach()
    theta_scaled[9] = theta_scaled[9] * float(scale)
    return theta_scaled


def phase16_scale_diffusion_path(theta_base, scale):
    theta_scaled = theta_base.clone().detach()
    theta_scaled[8] = theta_scaled[8] * float(scale)
    return theta_scaled


def phase16_run_from_theta(theta_init, family, label, checkpoints=phase16_checkpoints):
    model = MinimalSymNet().to(device)
    phase16_assign_theta(model, theta_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=random_frozen_lr)
    initial_coeffs = product_term_dict(model)
    initial_coeff_vector = coeff_vector_from_dict(initial_coeffs, ordered_names=phase16_feature_names)
    history_rows = []
    checkpoint_rows = []

    for epoch in range(random_frozen_epochs + 1):
        pred_ut = model(frozen_feature_tensor)
        pde_loss = random_frozen_loss_fn(pred_ut, frozen_ut_target)
        coeffs = product_term_dict(model)
        coeff_vector = coeff_vector_from_dict(coeffs, ordered_names=phase16_feature_names)
        row = {
            "family": family,
            "label": label,
            "epoch": epoch,
            "pde_mse": float(pde_loss.item()),
            "coeff(u*u_x)": float(coeffs["u*u_x"]),
            "coeff(u_xx)": float(coeffs["u_xx"]),
            "||xi - xi_LS||_2": float(np.linalg.norm(coeff_vector - xi_LS)),
            "||xi - xi_true||_2": float(np.linalg.norm(coeff_vector - xi_true)),
        }
        history_rows.append(row)
        if epoch in checkpoints:
            checkpoint_rows.append(row.copy())

        if epoch == random_frozen_epochs:
            break

        optimizer.zero_grad()
        pde_loss.backward()
        optimizer.step()

    final_coeffs = product_term_dict(model)
    final_coeff_vector = coeff_vector_from_dict(final_coeffs, ordered_names=phase16_feature_names)
    final_pde_mse = float(history_rows[-1]["pde_mse"])
    final_xi_dist = float(np.linalg.norm(final_coeff_vector - xi_LS))

    return {
        "family": family,
        "label": label,
        "theta_init": torch.as_tensor(theta_init, dtype=torch.float64).clone(),
        "initial_coeffs": initial_coeffs,
        "initial_coeff_vector": initial_coeff_vector,
        "final_coeffs": final_coeffs,
        "final_coeff_vector": final_coeff_vector,
        "history_df": pd.DataFrame(history_rows),
        "trajectory_df": pd.DataFrame(checkpoint_rows),
        "initial_pde_mse": float(history_rows[0]["pde_mse"]),
        "final_pde_mse": final_pde_mse,
        "final_xi_dist_to_ls": final_xi_dist,
        "returned_near_good_basin": bool((final_pde_mse <= phase16_good_pde_threshold) and (final_xi_dist <= phase16_good_xi_threshold)),
    }


phase16_runs = []

phase16_interp_alphas = [round(value, 1) for value in np.linspace(0.0, 1.0, 11)]
for alpha in phase16_interp_alphas:
    theta_alpha = (1.0 - alpha) * theta_good + alpha * theta_bad
    run = phase16_run_from_theta(theta_alpha, family="interp_good_to_bad", label=f"alpha={alpha:.1f}")
    run["alpha"] = alpha
    phase16_runs.append(run)

phase16_product_scales = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
for scale in phase16_product_scales:
    theta_scale = phase16_scale_product_path(theta_good, scale)
    run = phase16_run_from_theta(theta_scale, family="weaken_product_path", label=f"product_scale={scale:.2f}")
    run["product_scale"] = scale
    phase16_runs.append(run)

phase16_diffusion_scales = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
for scale in phase16_diffusion_scales:
    theta_scale = phase16_scale_diffusion_path(theta_good, scale)
    run = phase16_run_from_theta(theta_scale, family="weaken_diffusion", label=f"diffusion_scale={scale:.2f}")
    run["diffusion_scale"] = scale
    phase16_runs.append(run)

phase16_wrong_case_thetas = {
    "+1.0*u*u_x + 0.02*u_xx": phase16_theta_from_equation(+1.0, +NU_TRUE),
    "-0.5*u*u_x + 0.02*u_xx": phase16_theta_from_equation(-0.5, +NU_TRUE),
    "-0.25*u*u_x + 0.02*u_xx": phase16_theta_from_equation(-0.25, +NU_TRUE),
    "-1.0*u*u_x - 0.02*u_xx": phase16_theta_from_equation(-1.0, -NU_TRUE),
    "-1.0*u*u_x + 0.00*u_xx": phase16_theta_from_equation(-1.0, 0.0),
}
for label, theta_case in phase16_wrong_case_thetas.items():
    run = phase16_run_from_theta(theta_case, family="wrong_sign_or_magnitude", label=label)
    phase16_runs.append(run)

phase16_summary_rows = []
phase16_trajectory_rows = []
for run in phase16_runs:
    phase16_summary_rows.append({
        "family": run["family"],
        "initialization label": run["label"],
        "initial PDE MSE": run["initial_pde_mse"],
        "final PDE MSE": run["final_pde_mse"],
        "initial coeff(u*u_x)": float(run["initial_coeffs"]["u*u_x"]),
        "final coeff(u*u_x)": float(run["final_coeffs"]["u*u_x"]),
        "initial coeff(u_xx)": float(run["initial_coeffs"]["u_xx"]),
        "final coeff(u_xx)": float(run["final_coeffs"]["u_xx"]),
        "final ||xi - xi_LS||_2": run["final_xi_dist_to_ls"],
        "returned near good basin": run["returned_near_good_basin"],
    })
    phase16_trajectory_rows.extend(run["trajectory_df"].to_dict("records"))

phase16_summary_df = pd.DataFrame(phase16_summary_rows)
phase16_trajectory_df = pd.DataFrame(phase16_trajectory_rows)
display(phase16_summary_df)

phase16_family_tables = {}
for family_name in phase16_summary_df["family"].unique():
    phase16_family_tables[family_name] = phase16_summary_df[phase16_summary_df["family"] == family_name].reset_index(drop=True)
    display(phase16_family_tables[family_name])

phase16_family_return_df = pd.DataFrame([
    {
        "family": family_name,
        "returned near good basin": int(group["returned near good basin"].sum()),
        "total runs": int(len(group)),
        "median final PDE MSE": float(group["final PDE MSE"].median()),
        "median final ||xi - xi_LS||_2": float(group["final ||xi - xi_LS||_2"].median()),
    }
    for family_name, group in phase16_summary_df.groupby("family", sort=False)
])
display(phase16_family_return_df)

phase16_interp_df = phase16_family_tables["interp_good_to_bad"].copy()
phase16_interp_good = phase16_interp_df[phase16_interp_df["returned near good basin"]]
phase16_interp_bad = phase16_interp_df[~phase16_interp_df["returned near good basin"]]

print("Phase 16 setup")
print(f"- theta_good reused from the earlier Burgers-like frozen-SymNet point: PDE MSE = {ref_loss_value:.6e}")
print(f"- theta_bad reused from the known bad random-final frozen-SymNet point: PDE MSE = {float(random_frozen_summary['final_pde_mse']):.6e}")
print(f"- xi_true = true Burgers coefficients, xi_LS = exact Phase 11 least-squares coefficients on the frozen surrogate field")
print(f"- good-basin return rule reused from Phase 15: final PDE MSE <= {phase16_good_pde_threshold:.6e} and final ||xi - xi_LS||_2 <= {phase16_good_xi_threshold:.6e}")
print()
print("Interpolation family summary:")
print(phase16_interp_df[[
    "initialization label",
    "initial PDE MSE",
    "final PDE MSE",
    "initial coeff(u*u_x)",
    "final coeff(u*u_x)",
    "initial coeff(u_xx)",
    "final coeff(u_xx)",
    "final ||xi - xi_LS||_2",
    "returned near good basin",
]].to_string(index=False))
print()
print("Product-path weakening family summary:")
print(phase16_family_tables["weaken_product_path"][[
    "initialization label",
    "initial PDE MSE",
    "final PDE MSE",
    "initial coeff(u*u_x)",
    "final coeff(u*u_x)",
    "initial coeff(u_xx)",
    "final coeff(u_xx)",
    "final ||xi - xi_LS||_2",
]].to_string(index=False))
print()
print("Diffusion weakening family summary:")
print(phase16_family_tables["weaken_diffusion"][[
    "initialization label",
    "initial PDE MSE",
    "final PDE MSE",
    "initial coeff(u*u_x)",
    "final coeff(u*u_x)",
    "initial coeff(u_xx)",
    "final coeff(u_xx)",
    "final ||xi - xi_LS||_2",
]].to_string(index=False))
print()
print("Wrong-sign / wrong-magnitude family summary:")
print(phase16_family_tables["wrong_sign_or_magnitude"][[
    "initialization label",
    "initial PDE MSE",
    "final PDE MSE",
    "initial coeff(u*u_x)",
    "final coeff(u*u_x)",
    "initial coeff(u_xx)",
    "final coeff(u_xx)",
    "final ||xi - xi_LS||_2",
]].to_string(index=False))


def phase16_plot_family_trajectories(ax, family_name, color_values=None):
    family_runs = [run for run in phase16_runs if run["family"] == family_name]
    cmap = plt.cm.viridis
    colors = color_values if color_values is not None else np.linspace(0.1, 0.9, len(family_runs))
    for idx, run in enumerate(family_runs):
        traj = run["trajectory_df"]
        color = cmap(colors[idx]) if np.ndim(colors) else cmap(colors)
        ax.plot(
            traj["coeff(u*u_x)"],
            traj["coeff(u_xx)"],
            "-o",
            linewidth=1.8,
            markersize=4,
            color=color,
            alpha=0.95,
            label=run["label"],
        )
        ax.scatter(traj.iloc[0]["coeff(u*u_x)"], traj.iloc[0]["coeff(u_xx)"], color=color, marker="s", s=30)
        ax.scatter(traj.iloc[-1]["coeff(u*u_x)"], traj.iloc[-1]["coeff(u_xx)"], color=color, marker="o", s=40, edgecolor="black", linewidth=0.4)
    ax.scatter([-1.0], [NU_TRUE], color="black", marker="*", s=110, label="true Burgers")
    ax.scatter([xi_LS[4]], [xi_LS[2]], color="red", marker="X", s=90, label="Phase 11 xi_LS")
    ax.set_xlabel("coeff(u*u_x)")
    ax.set_ylabel("coeff(u_xx)")
    ax.set_title(family_name.replace('_', ' '))


fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=False, sharey=False)
phase16_plot_family_trajectories(axes[0, 0], "interp_good_to_bad", color_values=np.linspace(0.0, 1.0, len([run for run in phase16_runs if run["family"] == "interp_good_to_bad"])))
phase16_plot_family_trajectories(axes[0, 1], "weaken_product_path")
phase16_plot_family_trajectories(axes[1, 0], "weaken_diffusion")
phase16_plot_family_trajectories(axes[1, 1], "wrong_sign_or_magnitude")
axes[0, 0].legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=False)
for ax, family_name in zip(axes.ravel(), phase16_summary_df["family"].unique()):
    family_runs = [run for run in phase16_runs if run["family"] == family_name]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(family_runs)))
    for color, run in zip(colors, family_runs):
        hist = run["history_df"]
        ax.plot(hist["epoch"], hist["||xi - xi_LS||_2"], linewidth=1.8, color=color, alpha=0.9, label=run["label"])
    ax.set_yscale("log")
    ax.set_title(f"{family_name.replace('_', ' ')} distance to xi_LS")
    ax.set_xlabel("epoch")
    ax.set_ylabel("||xi - xi_LS||_2")
axes[0, 0].legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

phase16_failed_runs_df = phase16_summary_df[~phase16_summary_df["returned near good basin"]].copy()
phase16_interpolation_failed_span = pd.DataFrame([
    {
        "quantity": "alpha range failing to return",
        "value": f"{phase16_interp_bad['initialization label'].iloc[0]} through {phase16_interp_bad['initialization label'].iloc[-1]}",
    },
    {
        "quantity": "number of interpolation starts returning",
        "value": f"{len(phase16_interp_good)} / {len(phase16_interp_df)}",
    },
    {
        "quantity": "number of non-interpolation starts returning",
        "value": f"{int(phase16_summary_df[phase16_summary_df['family'] != 'interp_good_to_bad']['returned near good basin'].sum())} / {int((phase16_summary_df['family'] != 'interp_good_to_bad').sum())}",
    },
])
display(phase16_interpolation_failed_span)
display(phase16_failed_runs_df)
"""


PHASE16_INTERPRETATION = """
### Section 16 Interpretation

On September 3, 2026, the controlled initialization map showed that initialization matters strongly, but not in the simple form of a single contiguous interval along the straight line from `theta_good` to `theta_bad`. Along that specific interpolation in parameter space, starts at `alpha = 0.0`, `0.1`, `0.8`, and `0.9` returned to the good Burgers-like / least-squares basin under the same return rule used in Phase 15, while starts at `alpha = 0.2` through `0.7` and the exact `theta_bad` endpoint did not. So a visible basin split does appear, but the split is non-monotone in `alpha`, which is consistent with the earlier warning that straight lines in `theta` do not map cleanly to straight lines in coefficient space and can cut across SymNet redundancies and warped coordinates.

The clearest controlled failure mode came from that interpolation family. Its failed runs formed a systematic band of suppressed-transport, weakened-diffusion final states rather than recovering the Burgers-like point: representative finals ranged from about `(-0.764664, 0.008818)` at `alpha = 0.2` through `(-0.003324, 0.002751)` at `alpha = 0.7`, with the exact bad endpoint ending near `(-0.040230, -0.004196)`. That is not one perfectly unique bad point, but it is also not arbitrary scatter: failed runs drift toward a related low-transport region with much smaller `|coeff(u*u_x)|`, smaller `coeff(u_xx)`, and much larger `||xi - xi_LS||_2` than the good basin.

The other three families were much more forgiving than the interpolation family. Weakening only the product pathway by scaling the product readout, even all the way to `product_scale = 0.0`, still recovered the good basin after training in every tested case, though the smallest product scales slightly worsened final PDE MSE. Weakening only the linear `u_xx` pathway was even less fragile: every tested diffusion scale from `1.0` down to `0.0` returned to essentially the same final Burgers-like / LS-like point. The wrong-sign and wrong-magnitude starts were also corrected robustly. Even `+1.0 * u*u_x + 0.02 * u_xx` and `-1.0 * u*u_x - 0.02 * u_xx` converged back to the same good basin on this frozen dataset. So in this local diagnostic, incorrect sign by itself is not the dominant obstacle once the initialization remains on the simple hand-constructed Burgers support.

The most informative visualization is the coefficient-plane trajectory plot for the interpolation family. It shows two practical behaviors under identical training dynamics: one group of starts moves into the same compact region near the Phase 11 least-squares point, while another group bends toward a low-transport, low-diffusion band instead. The conservative conclusion is therefore that initialization controls SymNet recovery strongly and systematically, but the decisive variable is not merely the scalar size or sign of the Burgers coefficients. What matters more is whether the full parameter-space initialization lands in the region connected to the good Burgers-like support. The next open question is how to parameterize or initialize SymNet so that coefficient-aligned starts are reachable from generic initialization without relying on hand construction.
"""


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cells = notebook["cells"]

    new_question = "10. How does frozen SymNet convergence depend on structured initialization families?"
    intro = "".join(cells[0]["source"])
    if new_question not in intro:
        intro = intro.rstrip() + "\n10. How does frozen SymNet convergence depend on structured initialization families?\n"
        cells[0]["source"] = source_lines(intro)

    phase16_title = "## 16. Controlled Initialization and Convergence Map"
    filtered_cells = []
    skip = False
    for cell in cells:
        src = "".join(cell.get("source", []))
        if src.startswith(phase16_title):
            skip = True
            continue
        if skip and src.startswith("### Section 16 Interpretation"):
            skip = False
            continue
        if skip:
            continue
        filtered_cells.append(cell)

    filtered_cells.extend([
        markdown_cell(PHASE16_MARKDOWN),
        code_cell(PHASE16_CODE),
        markdown_cell(PHASE16_INTERPRETATION),
    ])

    notebook["cells"] = filtered_cells
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
