# Clean-baseline endpoint investigation

Post-hoc descriptive checks; no additional optimization or least-squares extraction.
Late changes and threshold occupancy use the last 20000 optimizer steps. True-coefficient residuals substitute the physical targets into saved learned fields; they do not fix coefficients during training.

Seed 0: endpoint strong; absolute coefficient errors (0.085867, 0.00261112); late coefficient changes (-0.0289997, -0.000643473); late strong fraction 0.5995.
Seed 1: endpoint strong; absolute coefficient errors (0.0779451, 0.00100232); late coefficient changes (-0.0217736, -3.13167e-05); late strong fraction 0.9798.

These observations cannot identify a unique optimization failure mechanism. Five repeats estimate variability only on one fixed numerical initial condition and grid.
