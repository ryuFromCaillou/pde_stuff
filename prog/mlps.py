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
        self.linears = nn.ModuleList([nn.Linear(in_dim+i, prod_dim, bias=False) for i in range(num_layers)])
        
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
        if self.linear.out_features != 2:
            raise ValueError("Only defined for prod_dim=2 in this EQL form.")
        A = self.linear.weight          # (2, N)
        w_p = self.readout.weight[0, -1]  # scalar (product term weight from readout layer)
        M = w_p * torch.outer(A[0], A[1])  # (N, N)
        return 0.5 * (M + M.T) if symmetrize else M

