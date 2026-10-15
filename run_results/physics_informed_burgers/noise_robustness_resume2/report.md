# Burgers observation-noise robustness

Two initialization seeds (0, 1) provide exploratory evidence only. Noise is applied once to the 640 training targets; all evaluation references are clean.

|sigma|seed|transport|diffusion|transport error|diffusion error|field rel L2|ut rel L2|first loose|first strong|final|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0|0|-0.914133|0.02261112|0.08586705|0.002611115|0.004997356|0.1048643|53773|87997|strong|
|0|1|-0.9220549|0.02100232|0.07794505|0.001002321|0.005197876|0.09125031|33634|79807|strong|
|0.01|0|-0.9238742|0.02064902|0.0761258|0.0006490159|0.006122974|0.0896356|53808|83823|strong|
|0.01|1|-0.9137351|0.01719612|0.08626491|0.002803883|0.005081063|0.09177051|35631|89708|strong|
|0.02|0|-0.9235682|0.01876725|0.07643181|0.001232747|0.006821541|0.0881101|56742|87304|strong|
|0.02|1|-0.893941|0.0130523|0.106059|0.006947699|0.007928153|0.1097704|38943|None|loose|
|0.05|0|-0.9007736|0.01051652|0.09922642|0.009483479|0.01080115|0.1054808|60171|99500|strong|
|0.05|1|-0.8477075|0.002900548|0.1522925|0.01709945|0.01256928|0.1714761|50803|None|loose|
|0.1|0|-0.8490754|-0.005059506|0.1509246|0.02505951|0.02282007|0.1775838|None|None|not_recovered|
|0.1|1|-0.7737396|-0.006572345|0.2262604|0.02657235|0.02637245|0.2843068|None|None|not_recovered|
|0.2|0|-0.7771989|-0.002973973|0.2228011|0.02297397|0.04525547|0.3019997|None|None|not_recovered|
|0.2|1|-0.663055|-0.002036084|0.336945|0.02203608|0.04870021|0.4552617|None|None|not_recovered|

The noisy training observation MSE and independent clean-reference residual are reported separately. PDE residuals use surrogate autodiff derivatives and the physical head; they never use finite differences of noisy observations. Do not interpret two-seed outcomes as recovery probabilities. Sampling sparsity and crossed noise-sampling experiments were not run.
