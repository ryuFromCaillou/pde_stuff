# Burgers sampling robustness

Two initialization seeds provide exploratory evidence only. Ratio 1.00 reuses the validated clean 640-observation control; the other four ratios are newly trained nested subsets.

|ratio|count|seed|transport|diffusion|transport error|diffusion error|field rel L2|ut rel L2|first loose|first strong|final|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|640|0|-0.914133|0.02261112|0.08586705|0.002611115|0.004997356|0.1048643|53773|87997|strong|
|1|640|1|-0.9220549|0.02100232|0.07794505|0.001002321|0.005197876|0.09125031|33634|79807|strong|
|0.75|480|0|-0.8732691|0.01685103|0.1267309|0.003148968|0.007570919|0.1339833|57169|None|loose|
|0.75|480|1|-0.9195535|0.02420577|0.08044654|0.004205768|0.006449509|0.1018188|38542|78921|strong|
|0.5|320|0|-0.8798308|0.01893562|0.1201692|0.001064381|0.006956215|0.1339605|67257|None|loose|
|0.5|320|1|-0.9251229|0.01976955|0.07487714|0.0002304506|0.004461196|0.08773348|38332|74101|strong|
|0.25|160|0|-0.812543|0.01746533|0.187457|0.002534667|0.00809167|0.1980135|88227|None|loose|
|0.25|160|1|-0.8727935|0.01378181|0.1272065|0.006218189|0.005579028|0.1291131|55370|None|loose|
|0.1|64|0|-0.8834904|0.01633031|0.1165096|0.003669687|0.01855573|0.218003|72897|None|loose|
|0.1|64|1|-0.8230436|0.006338851|0.1769564|0.01366115|0.03711799|0.4877377|75697|None|loose|

Sampling is a nested prefix of one reproducible permutation of the original 640 coordinates. Minibatches sample with replacement and use batch size min(640,count). The clean 51×256 evaluation grid and solver are unchanged. PDE residuals use surrogate autodiff derivatives; no noise, L1, SymNet, or least-squares extraction was used.

Noise and crossed noise–sampling experiments were not run.
