# Deliberation: Why Rusanov vs spectral?
**Pros:** shock capturing, straightforward, stable under CFL with flux splitting.  
**Cons:** numerical diffusion; spectral gives high-order pre-shock but Gibbs post-shock.  
**Decision:** start with Rusanov, later compare to WENO/spectral.
