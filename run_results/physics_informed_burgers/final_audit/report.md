# Clean-baseline endpoint investigation

Post-hoc descriptive checks; no additional optimization or least-squares extraction.
Late changes and threshold occupancy use the last 20000 optimizer steps. True-coefficient residuals substitute the physical targets into saved learned fields; they do not fix coefficients during training.

Seed 0: endpoint strong; absolute coefficient errors (0.085867, 0.00261112); late coefficient changes (-0.0289997, -0.000643473); late strong fraction 0.5995.
Seed 1: endpoint strong; absolute coefficient errors (0.0779451, 0.00100232); late coefficient changes (-0.0217736, -3.13167e-05); late strong fraction 0.9798.
Seed 2: endpoint strong; absolute coefficient errors (0.0986227, 0.00713715); late coefficient changes (-0.0263126, -5.08949e-05); late strong fraction 0.0587.
Seed 3: endpoint strong; absolute coefficient errors (0.0886288, 0.00356916); late coefficient changes (-0.0342739, 0.000908645); late strong fraction 0.3863.
Seed 4: endpoint loose; absolute coefficient errors (0.116593, 0.000745155); late coefficient changes (-0.0310931, -0.000297654); late strong fraction 0.0000.
Seed 5: endpoint strong; absolute coefficient errors (0.0810018, 0.00108824); late coefficient changes (-0.0345564, 0.00130953); late strong fraction 0.5993.

Strong clean recovery is not uniform within the declared horizon. The saved late changes distinguish continued progress from an approximately stationary endpoint; they do not establish eventual recovery. Compare remaining temporal/spatial derivative errors and the true-coefficient residual with the learned-head residual in endpoint_diagnostics.csv. Low learned residual alone cannot certify the physical coefficients. Do not proceed to noise or sparsity or tune failed seeds independently. A matched all-seed horizon or objective investigation requires a separately reviewed protocol.

These observations cannot identify a unique optimization failure mechanism. Five repeats estimate variability only on one fixed numerical initial condition and grid.
