# PDE Extraction with `mlps.py`

This note shows how to adapt `prog/mlps.py` for PDE discovery in a way that is similar in spirit to PDE-Net: learn a neural representation, build candidate PDE terms from derivatives / nonlinear products, and then read out a symbolic equation from the trained parameters.

## 1. What `mlps.py` already gives us

`mlps.py` contains two useful building blocks:

- `SirenMLP`: a smooth function approximator for scalar fields `u(t, x)`.
- `EQL`: a small symbolic-style network that creates product neurons from linear combinations of input features, which is useful for discovering sparse algebraic structure.

That means the file already supports the two main ingredients we need for PDE discovery:

1. **Approximate the solution field** with a neural model.
2. **Construct candidate PDE features** from learned or computed quantities.

## 2. PDE discovery pipeline

A PDE discovery workflow can be organized like this:

### Step A: Fit a smooth surrogate for the solution

Use `SirenMLP` to fit the observed data `u(t, x)`.

Why this helps:

- SIREN-style sine activations are smooth.
- Smoothness makes automatic differentiation stable.
- We can evaluate `u`, `u_t`, `u_x`, `u_xx`, etc. at arbitrary collocation points.

### Step B: Build a feature library

At sampled points, construct a feature vector such as:

- `u`
- `u_x`
- `u_xx`
- `u^2`
- `u u_x`
- `u^3`
- higher-order or mixed terms if needed

For systems, include features for each channel and cross-channel products.

### Step C: Fit the PDE right-hand side

Train a sparse model that maps those features to the time derivative:

- target: `u_t`
- input: candidate feature library
- model: linear model, sparse regression, or `EQL`

This is the analogue of PDE-Net’s symbolic head.

### Step D: Extract the symbolic PDE

After training, inspect the learned weights:

- keep terms whose coefficients are above a threshold
- map each feature index back to its symbolic name
- optionally simplify / prune small terms

The output is a readable equation like:

`u_t = c1 u_xx + c2 u u_x + c3 u`

## 3. How this maps to PDE-Net ideas

PDE-Net learns PDE structure through learned finite-difference filters plus a symbolic polynomial network. In `mlps.py`, the same idea can be implemented more directly:

- **PDE-Net derivative filters**  
  ? automatic differentiation on a smooth surrogate (`SirenMLP`)

- **PDE-Net SymNet / polynomial readout**  
  ? `EQL` or a sparse linear head over a feature library

- **Coefficient extraction**  
  ? inspect `EQL` weights and drop small coefficients

So the core concept is the same: learn a smooth representation first, then symbolically read off the governing equation.

## 4. Suggested implementation pattern

A practical implementation inside `mlps.py` would look like this:

### 4.1 Train the surrogate

```python
model = SirenMLP(hidden_size=64, hidden_layers=4)
```

Train it on observed `(t, x) -> u` samples.

### 4.2 Differentiate the surrogate

At collocation points:

```python
u = model(t, x)
u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
```

### 4.3 Build the feature tensor

For example:

```python
features = torch.cat([u, u_x, u_xx, u * u_x, u ** 2], dim=1)
```

### 4.4 Fit sparse discovery head

Use `EQL` or a simpler sparse linear model:

```python
discovery = EQL(in_dim=features.shape[1], prod_dim=2, num_layers=1)
pred = discovery(features)
```

If you want the cleanest PDE form, a linear sparse regressor is often easiest to interpret. `EQL` is useful when you want richer nonlinear combinations.

## 5. Reading out the equation

A simple extraction rule is:

1. assign names to each feature column
2. examine learned coefficients
3. sort by magnitude
4. keep terms above a tolerance

Example mapping:

- `feature[0] -> u`
- `feature[1] -> u_x`
- `feature[2] -> u_xx`
- `feature[3] -> u u_x`

Then a weight vector like `[0.00, 0.02, 0.11, -0.50]` becomes:

`u_t = 0.11 u_xx - 0.50 u u_x`

## 6. Where `EQL` fits best

The current `EQL` module is a good starting point if you want to discover symbolic structure beyond plain linear sparsity.

It already does three useful things:

- creates learned product neurons
- appends them back into the feature state
- stores intermediate tensors for inspection

That makes it suitable for discovering terms that are polynomial in the PDE library, especially if you want to emulate PDE-Net’s symbolic polynomial head.

## 7. Recommended extension for PDE discovery

If we want `mlps.py` to become a full PDE discovery module, the best next step is to add a small wrapper class such as `PDEDiscoveryNet` that combines:

- `SirenMLP` for smooth interpolation
- autodiff feature generation
- `EQL` or sparse readout
- a `coeffs()` method that returns symbolic terms and coefficients

A sketch of the API:

```python
class PDEDiscoveryNet(nn.Module):
    def forward(self, t, x):
        ...
    def library(self, t, x):
        ...
    def coeffs(self, threshold=1e-6):
        ...
```

That mirrors PDE-Net’s pattern:

- forward model
- symbolic expression
- coefficient extraction

## 8. Practical notes

- Use normalized inputs `t, x` for stable SIREN training.
- Prefer collocation points with good coverage of the spatiotemporal domain.
- If the learned PDE is noisy, raise sparsity pressure or threshold small coefficients harder.
- If the field is not smooth enough, increase `hidden_layers` or `hidden_omega_0` cautiously.

## 9. Bottom line

`mlps.py` can support PDE discovery by turning the MLP into a smooth surrogate, differentiating it to form PDE candidate terms, and then using `EQL` or sparse regression to recover a symbolic equation. That is the same high-level idea as PDE-Net, just expressed with a smoother neural interpolant instead of finite-difference operator banks.
