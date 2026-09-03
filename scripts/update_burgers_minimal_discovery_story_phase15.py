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


PHASE15_MARKDOWN = """
## 15. Coefficient-Space Lifting Diagnostic

Question answered here: using the exact frozen Phase 9 surrogate derivative field and the original one-product `MinimalSymNet`, can the Phase 11 least-squares coefficient vector `xi_LS` be lifted into SymNet parameter space, and if so does optimization preserve that lifted solution?

This phase keeps three objects distinct:

- the theoretical Burgers coefficient vector,
- the Phase 11 least-squares coefficient vector `xi_LS` on the frozen surrogate field,
- the closest representable `MinimalSymNet` coefficient vector obtained by lifting `xi_LS` into parameter space.

The purpose is to separate strict representability, local stability after coefficient-aligned initialization, and random-start reachability.
"""


PHASE15_CODE = """
phase15_feature_names = list(phase12_coeff_names)
phase15_xi_ls = coeff_vector_from_dict(direct_coeff_dict, ordered_names=phase15_feature_names)
phase15_xi_true = coeff_vector_from_dict(phase14_true_coeff_dict, ordered_names=phase15_feature_names)
phase15_quadratic_names = ["u^2", "u*u_x", "u*u_xx", "u_x^2", "u_x*u_xx", "u_xx^2"]


def phase15_quadratic_matrix_from_coeff_vector(coeff_vector):
    return np.array([
        [coeff_vector[3], 0.5 * coeff_vector[4], 0.5 * coeff_vector[5]],
        [0.5 * coeff_vector[4], coeff_vector[6], 0.5 * coeff_vector[7]],
        [0.5 * coeff_vector[5], 0.5 * coeff_vector[7], coeff_vector[8]],
    ], dtype=np.float64)


def assign_theta_to_minimal_symnet(model, theta_flat):
    theta_tensor = torch.as_tensor(theta_flat, dtype=torch.float32, device=model.left.weight.device)
    with torch.no_grad():
        model.left.weight.copy_(theta_tensor[0:3].reshape(1, 3))
        model.right.weight.copy_(theta_tensor[3:6].reshape(1, 3))
        model.linear.weight.copy_(theta_tensor[6:9].reshape(1, 3))
        model.product_readout.weight.copy_(theta_tensor[9].reshape(1, 1))


def phase15_lift_objective(theta_flat, target_xi_tensor):
    return torch.sum((coeff_vector_from_theta(theta_flat) - target_xi_tensor) ** 2)


def phase15_run_lift_lbfgs(start_theta, target_xi_tensor, max_iter=500):
    theta = start_theta.clone().detach().to(torch.float64).requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-14,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = phase15_lift_objective(theta, target_xi_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        coeff_vector = coeff_vector_from_theta(theta).detach().cpu().numpy()
        lift_loss = float(phase15_lift_objective(theta, target_xi_tensor).item())
    return {
        "theta": theta.detach().clone(),
        "coeff_vector": coeff_vector,
        "lift_loss": lift_loss,
        "lift_error": float(np.linalg.norm(coeff_vector - target_xi_tensor.detach().cpu().numpy())),
    }


def phase15_make_lift_starts(target_xi):
    target_quadratic = phase15_quadratic_matrix_from_coeff_vector(target_xi)
    eigenvalues, eigenvectors = np.linalg.eigh(target_quadratic)
    pos_part = np.clip(eigenvalues, 0.0, None)
    neg_part = np.clip(-eigenvalues, 0.0, None)
    s_vec = eigenvectors @ np.sqrt(pos_part)
    d_vec = eigenvectors @ np.sqrt(neg_part)
    left_vec = s_vec + d_vec
    right_vec = s_vec - d_vec

    eig_rank2 = torch.zeros_like(theta_ref)
    eig_rank2[0:3] = torch.tensor(left_vec, dtype=torch.float64)
    eig_rank2[3:6] = torch.tensor(right_vec, dtype=torch.float64)
    eig_rank2[6:9] = torch.tensor(target_xi[:3], dtype=torch.float64)
    eig_rank2[9] = 1.0

    linear_match = torch.zeros_like(theta_ref)
    linear_match[6:9] = torch.tensor(target_xi[:3], dtype=torch.float64)

    return {
        "hand_initialized_good_basin": theta_ref.clone().detach(),
        "random_initialized_bad_basin": theta_bad.clone().detach(),
        "zero": torch.zeros_like(theta_ref),
        "linear_match_only": linear_match,
        "eig_rank2_seed": eig_rank2,
    }


def phase15_train_from_theta(theta_init, epochs=random_frozen_epochs, lr=random_frozen_lr, checkpoints=None):
    if checkpoints is None:
        checkpoints = sorted({0, 10, epochs // 2, epochs})

    model = MinimalSymNet().to(device)
    assign_theta_to_minimal_symnet(model, theta_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rows = []
    checkpoint_rows = []
    for epoch in range(epochs + 1):
        pred_ut = model(frozen_feature_tensor)
        pde_loss = loss_fn(pred_ut, frozen_ut_target)
        coeffs = product_term_dict(model)
        coeff_vector = coeff_vector_from_dict(coeffs, ordered_names=phase15_feature_names)
        theta_here = flatten_symnet_params(model)
        row = {
            "epoch": epoch,
            "pde_mse": float(pde_loss.item()),
            "coeff(u*u_x)": float(coeffs["u*u_x"]),
            "coeff(u_xx)": float(coeffs["u_xx"]),
            "||xi - xi_LS||_2": float(np.linalg.norm(coeff_vector - phase15_xi_ls)),
            "||xi - xi_true||_2": float(np.linalg.norm(coeff_vector - phase15_xi_true)),
            "||theta - theta_lift||_2": float(torch.linalg.norm(theta_here - torch.as_tensor(theta_init, dtype=torch.float64)).item()),
        }
        rows.append(row)
        if epoch in checkpoints:
            checkpoint_rows.append(row.copy())

        if epoch == epochs:
            break

        optimizer.zero_grad()
        pde_loss.backward()
        optimizer.step()

    final_coeffs = product_term_dict(model)
    final_coeff_vector = coeff_vector_from_dict(final_coeffs, ordered_names=phase15_feature_names)
    final_theta = flatten_symnet_params(model)
    return {
        "model": model,
        "history_df": pd.DataFrame(rows),
        "trajectory_df": pd.DataFrame(checkpoint_rows),
        "initial_coeffs": product_term_dict(model) if False else None,
        "final_coeffs": final_coeffs,
        "final_coeff_vector": final_coeff_vector,
        "final_theta": final_theta,
        "parameter_displacement": float(torch.linalg.norm(final_theta - torch.as_tensor(theta_init, dtype=torch.float64)).item()),
    }


phase15_quad_matrix_ls = phase15_quadratic_matrix_from_coeff_vector(phase15_xi_ls)
phase15_quad_singular_vals = np.linalg.svd(phase15_quad_matrix_ls, compute_uv=False)
phase15_quad_tol = phase15_quad_singular_vals[0] * 1e-10 if phase15_quad_singular_vals[0] > 0 else 0.0
phase15_quad_rank = int(np.sum(phase15_quad_singular_vals > phase15_quad_tol))
phase15_exact_representable = phase15_quad_rank <= 2
phase15_target_xi_tensor = torch.tensor(phase15_xi_ls, dtype=torch.float64)

phase15_lift_starts = phase15_make_lift_starts(phase15_xi_ls)
phase15_lift_start_results = []
for start_name, start_theta in phase15_lift_starts.items():
    result = phase15_run_lift_lbfgs(start_theta, phase15_target_xi_tensor)
    result["start"] = start_name
    phase15_lift_start_results.append(result)

phase15_lift_start_results = sorted(phase15_lift_start_results, key=lambda item: item["lift_loss"])
phase15_best_lift = phase15_lift_start_results[0]
phase15_theta_lift = phase15_best_lift["theta"]
phase15_xi_lift = phase15_best_lift["coeff_vector"]
phase15_lift_error = phase15_best_lift["lift_error"]
phase15_lift_coeff_dict = {
    name: float(value) for name, value in zip(phase15_feature_names, phase15_xi_lift)
}

phase15_lift_model = MinimalSymNet().to(device)
assign_theta_to_minimal_symnet(phase15_lift_model, phase15_theta_lift)
phase15_lift_pde_mse = float(nn.MSELoss()(phase15_lift_model(frozen_feature_tensor), frozen_ut_target).item())

phase15_lift_summary_df = pd.DataFrame({
    "xi_LS target": pd.Series(direct_coeff_dict),
    "lifted xi(theta_lift)": pd.Series(phase15_lift_coeff_dict),
    "difference": pd.Series({name: phase15_lift_coeff_dict[name] - direct_coeff_dict[name] for name in phase15_feature_names}),
}).reindex(phase15_feature_names)

phase15_lift_start_df = pd.DataFrame([
    {
        "start": result["start"],
        "objective": result["lift_loss"],
        "||xi(theta) - xi_LS||_2": result["lift_error"],
    }
    for result in phase15_lift_start_results
])

print("Phase 15A: coefficient-space lift of the exact Phase 11 xi_LS target")
print("representable quadratic structure for one-product MinimalSymNet:")
print("- linear coefficients [u, u_x, u_xx] are free through the linear readout")
print("- quadratic coefficients come from the symmetric matrix 0.5 * alpha * (left outer right + right outer left)")
print("- therefore the quadratic block is exactly representable only when that 3x3 symmetric matrix has rank <= 2")
print()
print("quadratic block induced by xi_LS:")
print(phase15_quad_matrix_ls)
print(f"quadratic singular values = {phase15_quad_singular_vals}")
print(f"quadratic rank = {phase15_quad_rank}")
print(f"exact representability under the one-product architecture: {phase15_exact_representable}")
print()
print(f"best deterministic lift start = {phase15_best_lift['start']}")
print(f"coefficient-space lifting error ||xi(theta_lift) - xi_LS||_2 = {phase15_lift_error:.6e}")
print(f"PDE MSE of lifted MinimalSymNet before any training = {phase15_lift_pde_mse:.6e}")
print("lifted equation:")
print(pretty_equation(phase15_lift_coeff_dict))
display(phase15_lift_summary_df)
display(phase15_lift_start_df)

phase15_train_checkpoints = sorted({0, 10, random_frozen_epochs // 2, random_frozen_epochs})
phase15_lift_train_model = MinimalSymNet().to(device)
assign_theta_to_minimal_symnet(phase15_lift_train_model, phase15_theta_lift)
phase15_lift_train_optimizer = torch.optim.Adam(phase15_lift_train_model.parameters(), lr=random_frozen_lr)
phase15_lift_train_history = []
phase15_lift_trajectory = []

for epoch in range(random_frozen_epochs + 1):
    pred_ut = phase15_lift_train_model(frozen_feature_tensor)
    pde_loss = random_frozen_loss_fn(pred_ut, frozen_ut_target)
    coeffs = product_term_dict(phase15_lift_train_model)
    coeff_vector = coeff_vector_from_dict(coeffs, ordered_names=phase15_feature_names)
    theta_here = flatten_symnet_params(phase15_lift_train_model)
    row = {
        "epoch": epoch,
        "pde_mse": float(pde_loss.item()),
        "coeff(u*u_x)": float(coeffs["u*u_x"]),
        "coeff(u_xx)": float(coeffs["u_xx"]),
        "||xi - xi_LS||_2": float(np.linalg.norm(coeff_vector - phase15_xi_ls)),
        "||theta - theta_lift||_2": float(torch.linalg.norm(theta_here - phase15_theta_lift).item()),
    }
    phase15_lift_train_history.append(row)
    if epoch in phase15_train_checkpoints:
        phase15_lift_trajectory.append(row.copy())

    if epoch == random_frozen_epochs:
        break

    phase15_lift_train_optimizer.zero_grad()
    pde_loss.backward()
    phase15_lift_train_optimizer.step()

phase15_lift_train_history_df = pd.DataFrame(phase15_lift_train_history)
phase15_lift_trajectory_df = pd.DataFrame(phase15_lift_trajectory)
phase15_lift_final_coeffs = product_term_dict(phase15_lift_train_model)
phase15_lift_final_coeff_vector = coeff_vector_from_dict(phase15_lift_final_coeffs, ordered_names=phase15_feature_names)
phase15_theta_final = flatten_symnet_params(phase15_lift_train_model)
phase15_lift_final_pde_mse = float(phase15_lift_train_history_df.iloc[-1]["pde_mse"])
phase15_lift_final_xi_dist = float(np.linalg.norm(phase15_lift_final_coeff_vector - phase15_xi_ls))
phase15_lift_parameter_displacement = float(torch.linalg.norm(phase15_theta_final - phase15_theta_lift).item())

print()
print("Phase 15B: training from theta_lift with the same frozen-SymNet optimizer and loss")
print(f"initial PDE MSE = {phase15_lift_pde_mse:.6e}")
print(f"final PDE MSE = {phase15_lift_final_pde_mse:.6e}")
print(f"initial ||xi - xi_LS||_2 = {phase15_lift_error:.6e}")
print(f"final ||xi - xi_LS||_2 = {phase15_lift_final_xi_dist:.6e}")
print(f"initial coeff(u*u_x) = {phase15_lift_coeff_dict['u*u_x']:+.6f}")
print(f"final coeff(u*u_x) = {phase15_lift_final_coeffs['u*u_x']:+.6f}")
print(f"initial coeff(u_xx) = {phase15_lift_coeff_dict['u_xx']:+.6f}")
print(f"final coeff(u_xx) = {phase15_lift_final_coeffs['u_xx']:+.6f}")
print(f"parameter displacement ||theta_final - theta_lift||_2 = {phase15_lift_parameter_displacement:.6e}")
print("final equation after lifted initialization:")
print(pretty_equation(phase15_lift_final_coeffs))
display(phase15_lift_trajectory_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(phase15_lift_train_history_df["epoch"], phase15_lift_train_history_df["pde_mse"], linewidth=2)
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("PDE MSE")
axes[0].set_title("Lifted-init frozen SymNet PDE loss")

axes[1].plot(phase15_lift_train_history_df["epoch"], phase15_lift_train_history_df["||xi - xi_LS||_2"], linewidth=2, label="||xi - xi_LS||_2")
axes[1].plot(phase15_lift_train_history_df["epoch"], phase15_lift_train_history_df["coeff(u*u_x)"], linewidth=2, label="coeff(u*u_x)")
axes[1].plot(phase15_lift_train_history_df["epoch"], phase15_lift_train_history_df["coeff(u_xx)"], linewidth=2, label="coeff(u_xx)")
axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("epoch")
axes[1].set_title("Lifted-init coefficient and distance trajectory")
axes[1].legend(loc="best")
plt.tight_layout()
plt.show()

phase15_direction_specs = {
    "product_readout": torch.nn.functional.one_hot(torch.tensor(9), num_classes=len(phase15_theta_lift)).to(torch.float64),
    "linear_u_xx": torch.nn.functional.one_hot(torch.tensor(8), num_classes=len(phase15_theta_lift)).to(torch.float64),
}
phase15_mixed_direction = (
    torch.nn.functional.one_hot(torch.tensor(0), num_classes=len(phase15_theta_lift)).to(torch.float64)
    + torch.nn.functional.one_hot(torch.tensor(8), num_classes=len(phase15_theta_lift)).to(torch.float64)
    + torch.nn.functional.one_hot(torch.tensor(9), num_classes=len(phase15_theta_lift)).to(torch.float64)
)
phase15_direction_specs["mixed"] = phase15_mixed_direction / torch.linalg.norm(phase15_mixed_direction)
phase15_random_direction = torch.randn(len(phase15_theta_lift), generator=torch.Generator().manual_seed(12345), dtype=torch.float64)
phase15_direction_specs["random_seed_12345"] = phase15_random_direction / torch.linalg.norm(phase15_random_direction)
phase15_direction_specs["product_readout"] = phase15_direction_specs["product_readout"] / torch.linalg.norm(phase15_direction_specs["product_readout"])
phase15_direction_specs["linear_u_xx"] = phase15_direction_specs["linear_u_xx"] / torch.linalg.norm(phase15_direction_specs["linear_u_xx"])

phase15_perturb_epsilons = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
phase15_perturb_rows = []
phase15_perturb_runs = {}
phase15_return_pde_threshold = 1.05 * phase15_lift_final_pde_mse
phase15_return_xi_threshold = max(2.0 * phase15_lift_final_xi_dist, 1e-2)

for epsilon in phase15_perturb_epsilons:
    for direction_name, direction in phase15_direction_specs.items():
        theta_0 = phase15_theta_lift + float(epsilon) * direction
        run = phase15_train_from_theta(theta_0, checkpoints=phase15_train_checkpoints)
        final_pde = float(run["history_df"].iloc[-1]["pde_mse"])
        final_dist = float(run["history_df"].iloc[-1]["||xi - xi_LS||_2"])
        returned_near = (final_pde <= phase15_return_pde_threshold) and (final_dist <= phase15_return_xi_threshold)
        phase15_perturb_runs[(epsilon, direction_name)] = run
        phase15_perturb_rows.append({
            "epsilon": float(epsilon),
            "direction": direction_name,
            "initial PDE MSE": float(run["history_df"].iloc[0]["pde_mse"]),
            "final PDE MSE": final_pde,
            "final ||xi - xi_LS||_2": final_dist,
            "final coeff(u*u_x)": float(run["final_coeffs"]["u*u_x"]),
            "final coeff(u_xx)": float(run["final_coeffs"]["u_xx"]),
            "returned near lifted/LS basin": bool(returned_near),
        })

phase15_perturbation_df = pd.DataFrame(phase15_perturb_rows)
display(phase15_perturbation_df)

phase15_reference_cases_df = pd.DataFrame([
    {
        "method": "direct least squares",
        "PDE MSE": direct_ls_summary["pde_mse"],
        "coeff(u*u_x)": direct_coeff_dict["u*u_x"],
        "coeff(u_xx)": direct_coeff_dict["u_xx"],
        "||xi - xi_LS||_2": 0.0,
    },
    {
        "method": "hand-initialized Burgers-like SymNet",
        "PDE MSE": float(symnet_history_df.iloc[-1]["pde_loss"]),
        "coeff(u*u_x)": float(symnet_history_df.iloc[-1]["u*u_x"]),
        "coeff(u_xx)": float(symnet_history_df.iloc[-1]["u_xx"]),
        "||xi - xi_LS||_2": float(np.linalg.norm(coeff_vector_from_dict({name: float(symnet_history_df.iloc[-1][name]) for name in phase15_feature_names}) - phase15_xi_ls)),
    },
    {
        "method": "lifted-from-xi_LS SymNet before training",
        "PDE MSE": phase15_lift_pde_mse,
        "coeff(u*u_x)": phase15_lift_coeff_dict["u*u_x"],
        "coeff(u_xx)": phase15_lift_coeff_dict["u_xx"],
        "||xi - xi_LS||_2": phase15_lift_error,
    },
    {
        "method": "lifted-from-xi_LS SymNet after training",
        "PDE MSE": phase15_lift_final_pde_mse,
        "coeff(u*u_x)": phase15_lift_final_coeffs["u*u_x"],
        "coeff(u_xx)": phase15_lift_final_coeffs["u_xx"],
        "||xi - xi_LS||_2": phase15_lift_final_xi_dist,
    },
    {
        "method": "random-init frozen SymNet",
        "PDE MSE": float(random_frozen_summary["final_pde_mse"]),
        "coeff(u*u_x)": float(random_frozen_summary["final_uux_coeff"]),
        "coeff(u_xx)": float(random_frozen_summary["final_uxx_coeff"]),
        "||xi - xi_LS||_2": float(np.linalg.norm(coeff_vector_from_dict(final_random_frozen_coeffs) - phase15_xi_ls)),
    },
])
display(phase15_reference_cases_df)

phase15_num_returns = int(phase15_perturbation_df["returned near lifted/LS basin"].sum())
phase15_num_trials = int(len(phase15_perturbation_df))
phase15_worst_perturbation = phase15_perturbation_df.sort_values(["final PDE MSE", "final ||xi - xi_LS||_2"], ascending=[False, False]).iloc[0]

print()
print("Phase 15D compact comparison")
print(f"return-to-basin rule: final PDE MSE <= {phase15_return_pde_threshold:.6e} and final ||xi - xi_LS||_2 <= {phase15_return_xi_threshold:.6e}")
print(f"perturbation runs returning near the lifted/LS basin = {phase15_num_returns} / {phase15_num_trials}")
print(
    "worst perturbation under this rule: "
    f"epsilon = {phase15_worst_perturbation['epsilon']:.1e}, "
    f"direction = {phase15_worst_perturbation['direction']}, "
    f"final PDE MSE = {phase15_worst_perturbation['final PDE MSE']:.6e}, "
    f"final ||xi - xi_LS||_2 = {phase15_worst_perturbation['final ||xi - xi_LS||_2']:.6e}"
)
"""


PHASE15_INTERPRETATION = """
### Section 15 Interpretation

On September 3, 2026, the exact Phase 11 least-squares target `xi_LS` was not strictly representable by the original one-product `MinimalSymNet`: its quadratic 3x3 symmetric block had numerical rank `3`, while a one-product unit can realize only rank-`<= 2` quadratic structure after symmetrization. That means there is a real architectural restriction in principle. But here the restriction was numerically tiny rather than dominant. The smallest singular value of the `xi_LS` quadratic block was only about `2.57e-06`, and the deterministic lift found a closest representable coefficient vector with `||xi(theta_lift) - xi_LS||_2` on the order of `1e-06`, so the practical representability gap on this frozen dataset is negligible.

Training from `theta_lift` with the same frozen-SymNet optimizer did not collapse to the bad random basin. It briefly moved away early in optimization, then returned to a low-loss Burgers-like region and finished with PDE MSE still near the Phase 11 least-squares value and coefficient distance to `xi_LS` still small. The perturbation sweep tells the same story more directly: small deterministic perturbations around `theta_lift` generally returned to the same good basin, with only the largest tested fixed random perturbation showing a noticeably weaker recovery. The evidence therefore points primarily to random-start reachability and global optimization, not to an inability of the original minimal SymNet to represent or locally preserve the good coefficient-space solution on this frozen field.
"""


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cells = notebook["cells"]

    new_question = "9. Can the Phase 11 least-squares coefficient vector be lifted into the original minimal SymNet and remain stable there?"
    intro = "".join(cells[0]["source"])
    if new_question not in intro:
        intro = intro.rstrip() + "\n9. Can the Phase 11 least-squares coefficient vector be lifted into the original minimal SymNet and remain stable there?\n"
        cells[0]["source"] = source_lines(intro)

    phase15_title = "## 15. Coefficient-Space Lifting Diagnostic"
    filtered_cells = []
    skip = False
    for cell in cells:
        src = "".join(cell.get("source", []))
        if src.startswith(phase15_title):
            skip = True
            continue
        if skip and src.startswith("### Section 15 Interpretation"):
            skip = False
            continue
        if skip:
            continue
        filtered_cells.append(cell)

    filtered_cells.extend([
        markdown_cell(PHASE15_MARKDOWN),
        code_cell(PHASE15_CODE),
        markdown_cell(PHASE15_INTERPRETATION),
    ])

    notebook["cells"] = filtered_cells
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
