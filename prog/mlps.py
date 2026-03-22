import torch
import torch.nn as nn

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class SimpleMLP(nn.Module):
    def __init__(self, n_layers, hidden_size, act=Sin):
        super(SimpleMLP, self).__init__()
        assert n_layers >= 2, "n_layers must be at least 2"

        layers = []
        # First layer: input (2,) -> hidden_size
        layers.append(nn.Linear(2, hidden_size))
        layers.append(act())

        # Hidden layers: (m -> m)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act())

        # Final layer: (m -> 1)
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        input_tensor = torch.cat((t, x), dim=1)  # Concatenate along feature axis
        return self.model(input_tensor)
    
class symMLP(nn.Module):
    """
    Readout-only baseline, EQL-swappable contract.

    v(F) = readout([F])

    - Keeps readout shape (K -> 1), bias=False
    - Simple linear transformation of features acted on by optimizer (GD, Lasso, etc.)

    """
    def __init__(self, in_dim: int, prod_dim: int = 2, bias: bool = False):
        super().__init__()
        self.readout = nn.Linear(in_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        Y = torch.cat([feats], dim=1)  # (N, K+1)
        return self.readout(Y)              # (N, 1)

class EQL(nn.Module):
    def __init__(self, in_dim, prod_dim=2, num_layers=1, bias=False):
        """
        in_dim: number of input features (e.g. u, ux, uxx)

        prod_dim: number of linear combinations to create for pairwise products
        """
        super().__init__()
        #
        #self.linear = nn.Linear(in_dim, prod_dim, bias=bias)
        self.readout = nn.Linear(in_dim + num_layers, 1, bias=False)
        self.linears = nn.ModuleList([nn.Linear(in_dim+i, prod_dim) for i in range(num_layers)])
        
    def forward(self, feats):
        for i, linear in enumerate(self.linears):
            Z = linear(feats)  # (N, prod_dim)
            self.preop_ns = Z  # save for inspection
            prod_neuron = torch.prod(Z, dim=1, keepdim=True)  # (N,1)
            self.postop_ns = prod_neuron  # save for inspection
            feats = torch.cat([feats, prod_neuron], dim=1)        # (N, in_dim+1)
            self.feats = feats  # save for inspection
        return self.readout(feats)
    
    @torch.no_grad()
    def effective_quadratic_matrix(self, symmetrize: bool = False):
        """
        Return the effective *quadratic* coefficient matrix in the original input-feature
        basis (i.e., coefficients on x_i x_j where x is the *initial* `feats` passed in).

        Notes
        -----
        - For `num_layers > 1`, this model is no longer purely quadratic in the original
          inputs; it contains higher-order terms (e.g. cubic/quartic/...). This method
          returns only the degree-2 part with respect to the original inputs.
        - Defined only for `prod_dim == 2`, where each product neuron is (a·f)(b·f).
        """
        if len(self.linears) == 0:
            raise ValueError("No product layers: effective quadratic matrix is undefined.")

        if any(l.out_features != 2 for l in self.linears):
            raise ValueError("Only defined for prod_dim=2 in this EQL form (all linears must have out_features==2).")

        K = self.linears[0].in_features  # original feature dimension
        if self.readout.in_features != K + len(self.linears):
            raise ValueError(
                f"readout.in_features should be {K + len(self.linears)} (in_dim + num_layers); "
                f"got {self.readout.in_features}."
            )

        w = self.readout.weight[0]  # (K + num_layers,)
        M = torch.zeros((K, K), device=w.device, dtype=w.dtype)

        # Each appended product neuron p_k contributes: w_prod_k * (a_k·x)(b_k·x) to the quadratic part,
        # where (a_k, b_k) are the first K entries of the corresponding linear layer rows.
        for k, linear in enumerate(self.linears):
            A = linear.weight  # (2, K + k)
            a0 = A[0, :K]
            a1 = A[1, :K]
            w_prod_k = w[K + k]
            M = M + w_prod_k * torch.outer(a0, a1)

        return 0.5 * (M + M.T) if symmetrize else M

