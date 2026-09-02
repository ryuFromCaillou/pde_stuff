# Research State

## Experiment objective
Generate a presentation-ready scatter plot of the actual burg_gen training coordinates after applying the active training stride.

## Exact configuration
```json
{
  "dataset_name": "burg_gen",
  "noise_level": 0.0,
  "seed": 0,
  "stride_t": 2,
  "stride_x": 2,
  "output_root": null,
  "output_tag": null
}
```

## Dataset
- dataset: `burg_gen`
- loader: `runs/run_eql_joint_training.py::load_dataset -> Datasets/data/processed/burg_gen/burg_gen.py::solve_burgers`
- coordinate space used for plotting: physical x/t coordinates

## Sampling summary
- original spatial grid size Nx: 256
- original temporal grid size Nt: 252
- stride_x: 2
- stride_t: 2
- sampled spatial count: 128
- sampled temporal count: 126
- total sampled (x,t) observations: 16128

## Artifact directory
`/home/ghost/ghost/pde_stuff/Datasets/data/processed/burg_gen/artifacts/training_coords_scatter_20260902T175741Z`

## Summary file
- summary: `/home/ghost/ghost/pde_stuff/Datasets/data/processed/burg_gen/artifacts/training_coords_scatter_20260902T175741Z/summary.json`
