# Datasets Agent

## Folder Purpose

Contains PDE dataset generators and dataset wrappers used by training
experiments.

This folder is responsible for producing:

    (t, x, y_clean, y_noisy)

pairs suitable for model fitting.

## Dataset Types

| Dataset | Entry Point (data/processed/<type>_gen/<type>_gen.py) |
|----------|-------------|
| Burgers | burg_gen.py |
| Allen-Cahn | allen_cahn_gen.py |
| Heat | heat_gen.py | 

## Dataset Contract

All dataset generators should return:

    t_s
    x_s
    y_s
    y_noisy
    N

where:

    t_s      = flattened time coordinates
    x_s      = flattened spatial coordinates
    y_s      = clean solution values
    y_noisy  = noisy solution values
    N        = spatial grid size

## Characterization Metrics

Every generated dataset should expose:

    t range
    x range
    Nt
    Nx
    total samples
    noise level

Optional experiment-specific metrics:

    stride_t
    stride_x
    effective samples

## Common Tasks

### Generate Burgers Dataset

Read:
    burg_gen.py

### Generate Allen-Cahn Dataset

Read:
    allen_cahn_gen.py

### Add New PDE

1. Create <pde>_gen.py
2. Follow Dataset Contract
3. Document characterization metrics
4. Add routing entry above

## Shared (share/)

This folder stores datasets shared from other repos. Folder structure looks like: 

Datasets/
├── shared/
│   ├── PDE-Net2/
│   │   ├── config.json
│   │   ├── burgers.py
│   │   ├── allen_cahn.py
|   |   ├── setplot.py
|   |   ├── scatter.png
│   │   └── AGENTS.md
│   │
│   ├── PINN-Benchmarks/
│   │   ├── config.json
│   │   ├── allen_cahn.npy
|   |   ├── setplot.py
|   |   ├── scatter.png
│   │   └── AGENTS.md
│   │
│   └── DeepMoD/
│       ├── config.json
|       ├── setplot.py
|       ├── scatter.png
│       └── burgers.npy

### config.json 
has keys like:
```json
{
    "repo": "PDE-Net2",
    "source": "github.com/xxx/pdenet2",
    "dataset_type": "generated",
    "coordinate_order": ["t","x"],
    "variables": ["u"],
    "default_params": {
        (if available else NA)
        "Nt": 201,
        "Nx": 256,
        "samples": 1000, 
        "noise_level": 0.05, 
        "noise_implementation": (some map on raw_data)
        (we can include physical constants like nu in burgers)
        v: 0.1, 
    },
    "plot_routes": {
        "quicklook": "setplot.field_2d",
        "default_value": "y_noisy",
        "fallback_value": "y_s",
        "plot_kind": "scatter"
    },
    "ts_shape": [1000],
    "xs_shape": [1000],
    "ys_shape": [1000]
    
}
```

### plot_routes

`plot_routes` tells the shared-dataset adapter how to visualize a dataset after it is converted into the standard contract:

- `quicklook` names the plotting helper to use, for example `setplot.field_2d`
- `default_value` picks the field shown by default, usually `y_noisy`
- `fallback_value` picks the clean field when noisy values are unavailable, usually `y_s`
- `plot_kind` describes the primary visualization style, usually `scatter` for subsampled shared datasets

The optional `ts_shape`, `xs_shape`, and `ys_shape` entries document the flattened tensor shapes expected by the adapter after conversion.

### setplot.field_2d

Assume the dataset has already been converted to the standard repo contract:

    t_s
    x_s
    y_s
    y_noisy
    N

Agent behavior:

1. Load the shared paper dataset through its adapter.
2. Receive standard outputs: `t_s, x_s, y_s, y_noisy, N`.
3. Plot `x_s` on the horizontal axis.
4. Plot `t_s` on the vertical axis.
5. Color by `y_noisy` by default.
6. Use `y_s` instead when the task asks for clean data.
7. Use scatter by default because shared paper datasets may be subsampled or irregular.
8. Use grid/heatmap only when `len(t_s) == Nt * Nx` and the data reshapes cleanly.

Default route:

```python
def field_2d(t_s, x_s, y_s, y_noisy, N, clean=False):
    values = y_s if clean else y_noisy

    fig, ax = plt.subplots()
    im = ax.scatter(x_s, t_s, c=values, marker="x", s=10)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, label="u_clean" if clean else "u_noisy")

    plt.show()
```

## Required Plot Artifact

When adapting any paper dataset under:

    Datasets/shared/<PaperID>/

the agent must generate a saved scatter plot image:

    Datasets/shared/<PaperID>/scatter.png

This is an artifact requirement, not only a plotting route.

The plot must use the standard adapted dataset contract:

    t_s, x_s, y_s, y_noisy, N

Plot rules:

    x-axis: x_s
    y-axis: t_s
    color: y_noisy by default
    marker: "x"
    output path: Datasets/shared/<PaperID>/scatter.png

If clean data is requested, also save:

    Datasets/shared/<PaperID>/scatter_clean.png

Required behavior:

```python
fig, ax = plt.subplots()
im = ax.scatter(x_s, t_s, c=y_noisy, marker="x", s=10)
ax.set_xlabel("x")
ax.set_ylabel("t")
fig.colorbar(im, ax=ax, label="u_noisy")
fig.savefig("Datasets/shared/<PaperID>/scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)

### DeepMoD Example

For `Datasets/shared/DeepMoD/burgers.npy`, the adapter can load the flattened arrays directly and pass them to `field_2d`:

```python
import numpy as np

data = np.load("Datasets/shared/DeepMoD/burgers.npy", allow_pickle=True).item()
field_2d(
    data["t_s"],
    data["x_s"],
    data["y_s"],
    data["y_noisy"],
    data["N"],
)
```

Use `clean=True` when the quicklook should render the clean field:

```python
field_2d(
    data["t_s"],
    data["x_s"],
    data["y_s"],
    data["y_noisy"],
    data["N"],
    clean=True,
)
```

### Other Repo Contracts

#### DeepMoD
(Path: ../Documents/DeePyMod-master)

DeepMoD loads a full coordinate grid through a dataset object, then applies noise, normalization, and optional random subsampling before returning coordinates and values. To fit the current contract, treat the DeepMoD output as the source of truth for the clean field, then flatten it into the repo format:

1. Load `coords` and `data` from the DeepMoD dataset.
2. Split the coordinate tensor into time and space columns, so `coords[:, 0] -> t_s` and `coords[:, 1] -> x_s`.
3. Use the pre-noise field as `y_s` and the noise-augmented field as `y_noisy`.
4. Keep the spatial resolution as `N` by using the original `Nx`/grid size before flattening.
5. Return all arrays as 1D float tensors/arrays in the order `t_s, x_s, y_s, y_noisy, N`.

Create arguments like
```python
v = 0.1
A = 1.0
x = torch.linspace(-3, 4, 100)
t = torch.linspace(0.5, 5.0, 50)
load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise_level": 0.05}
```

then pass into a dataset object that also handles noise, coordinate normalization, and random subsampling:

```python
dataset = Dataset(
    burgers_delta,
    load_kwargs=load_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 2000},
    device=device,
)
```

**Naming Convention:**
<data type>_<paper>, ex: BurgerSawtooth_DeepMod

