# Burgers sampling robustness

Two initialization seeds provide exploratory evidence only. Ratio 1.00 reuses the validated clean 640-observation control; the other four ratios are newly trained nested subsets.

|ratio|count|seed|transport|diffusion|transport error|diffusion error|field rel L2|ut rel L2|first loose|first strong|final|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|640|0|-0.914133|0.02261112|0.08586705|0.002611115|0.004997356|0.1048643|53773|87997|strong|
|1|640|1|-0.9220549|0.02100232|0.07794505|0.001002321|0.005197876|0.09125031|33634|79807|strong|
|0.75|480|0|0.002087958|0.002504894|1.002088|0.01749511|0.9691044|0.9761941|None|None|not_recovered|
|0.75|480|1|0.002430409|0.002485848|1.00243|0.01751415|1.099935|1.106996|None|None|not_recovered|
|0.5|320|0|0.002130535|0.002506324|1.002131|0.01749368|0.9692365|0.9762263|None|None|not_recovered|
|0.5|320|1|0.002469513|0.002468491|1.00247|0.01753151|1.102211|1.106701|None|None|not_recovered|
|0.25|160|0|0.00212005|0.002500138|1.00212|0.01749986|0.9642559|0.9763899|None|None|not_recovered|
|0.25|160|1|0.002459811|0.002453721|1.00246|0.01754628|1.073973|1.100139|None|None|not_recovered|
|0.1|64|0|0.002147778|0.002506947|1.002148|0.01749305|0.9707569|0.9763362|None|None|not_recovered|
|0.1|64|1|0.001129616|0.002460309|1.00113|0.01753969|1.097673|1.103096|None|None|not_recovered|

Sampling is a nested prefix of one reproducible permutation of the original 640 coordinates. Minibatches sample with replacement and use batch size min(640,count). The clean 51×256 evaluation grid and solver are unchanged. PDE residuals use surrogate autodiff derivatives; no noise, L1, SymNet, or least-squares extraction was used.

Noise and crossed noise–sampling experiments were not run.
