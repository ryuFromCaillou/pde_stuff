Agreed. For Tier 3, normalization is no longer just an implementation detail—it directly affects training dynamics. So it *should* be an axis in the experiment matrix.

Below is a Codex-ready `AGENTS.md` extension that adds normalization as an explicit test dimension for Tier 3.

---

````markdown
# Tier 3 Extension: Feature Normalization Ablation

## Objective

Evaluate the impact of feature normalization in joint PDE training.

This is a controlled ablation to determine whether `FeatureTensor(normalize=True)` improves:

- coefficient recovery
- training stability
- derivative consistency

compared to `normalize=False`.

This is now a formal axis in the experiment matrix.

---

# Updated Experiment Matrix

For each dataset and model, run:

Dataset:
- Burgers
- Allen–Cahn

Field model:
- SimpleMLP
- SIREN

Feature normalization:
- normalize = True
- normalize = False

PDE weight:
- λ_pde ∈ {0, 1e-3, 1e-2, 1e-1, 1}

Seeds:
- {0, 1, 2}

Total runs:
2 (datasets) × 2 (models) × 2 (normalize) × 5 (λ_pde) × 3 (seeds)

---

# Key Design Constraint

Normalization must only affect the feature library:

```python
feature_builder = FeatureTensor(
    terms=feature_terms,
    normalize=normalize_flag
).build
````

Do NOT change:

* dataset normalization
* input scaling
* training procedure

Only change `FeatureTensor(normalize=...)`.

---

# Expected Behavior

## normalize=True

* better conditioning
* more stable training
* smoother coefficient trajectories
* requires coefficient unscaling for physical interpretation

## normalize=False

* direct physical coefficients
* worse conditioning
* higher variance across seeds
* potential instability in joint training

---

# Required Implementation

## 1. Add normalize flag to experiment runner

In `run_pde_extraction_experiment.py` or Tier 3 runner:

Add loop:

```python
for normalize in [True, False]:
```

Pass into feature builder:

```python
feature_builder = FeatureTensor(
    terms=feature_terms,
    normalize=normalize
).build
```

Inject into trainer:

```python
trainer = PDETrainer(
    u_model,
    v_model,
    cfg,
    feature_builder=feature_builder
)
```

---

## 2. Log normalization setting

Each run must record:

```json
"normalize": true or false
```

Include in:

* JSON output
* CSV summary

---

## 3. Coefficient handling

### Case A: normalize=False

* coefficients are already physical
* compare directly to ground truth

### Case B: normalize=True

* must convert:

```python
w_phys = w_norm / scales
```

Where:

* `scales` comes from `FeatureTensorOut.scales`

Store BOTH:

* normalized coefficients
* physical coefficients

---

## 4. Logging additions

Each run should include:

* coeff_norm (raw from v_model)

* coeff_phys (after scaling if needed)

* coeff_error_phys

* coeff_error_norm (optional)

* training loss breakdown:

  * loss_data
  * loss_pde
  * l1

* coefficient trajectory over epochs

---

## 5. Output structure

```text
runs/tier3/
    burgers/
        normalize_true/
        normalize_false/
    allen_cahn/
        normalize_true/
        normalize_false/
```

Each subdirectory contains:

* per-seed runs
* summary.csv

---

# Evaluation Criteria

Compare normalize=True vs normalize=False on:

1. coefficient L2 error (physical)
2. coefficient variance across seeds
3. final data loss
4. PDE residual loss
5. coefficient stability over training

---

# Interpretation Targets

You are testing:

Does normalization improve PDE identification during joint training?

Specifically:

* does it stabilize gradients?
* does it reduce coefficient drift?
* does it improve recovery accuracy?

---

# Constraints

* Do not introduce EQL yet

* Use `symMLP` as PDE head

* Keep feature library fixed per dataset:

  * Burgers: ["u", "u_x", "u_xx", "uu_x"]
  * Allen–Cahn: ["u", "u_xx", "u3"]

* Keep all other hyperparameters fixed across normalize=True/False runs

---

# Validation Checklist

* Both normalize=True and normalize=False runs execute
* Coefficients are correctly unscaled when needed
* CSV includes normalize column
* No crashes from feature scaling differences
* Results are comparable across seeds

---

# Deliverables

1. normalization loop added to experiment runner
2. correct coefficient scaling logic implemented
3. results written to structured directories
4. summary CSV produced with normalization axis included

```


